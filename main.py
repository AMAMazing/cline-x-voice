from flask import Flask, jsonify, request, Response
# import webbrowser # Not used in this version
import win32clipboard
import time
import pywintypes
from time import sleep
# import os # Not directly used, but optimisewait might use it
from optimisewait import set_autopath, set_altpath # optimisewait itself not called
# import pyautogui # Not used in this version
import logging
import json
# from threading import Timer # Not used in this version
from typing import Union, List, Dict, Optional
import base64
import io
from PIL import Image
import re
from talktollm import talkto # Assuming this is correctly implemented elsewhere
import copy # Added for deepcopy
import sys # For custom print handler and stdout

# --- TTS Imports and Setup ---
import pyttsx3
from threading import Thread, Lock

tts_engine = None
tts_initialized_successfully = False
tts_lock = Lock()

def init_tts():
    global tts_engine, tts_initialized_successfully
    if tts_engine is not None:
        return
    try:
        logger.info("Initializing TTS engine...")
        tts_engine = pyttsx3.init()
        tts_initialized_successfully = True
        logger.info("TTS engine initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize TTS engine: {e}", exc_info=True)
        tts_engine = None
        tts_initialized_successfully = False

def speak_threaded(text_to_say: str):
    if not tts_initialized_successfully or tts_engine is None:
        logger.warning("TTS engine not initialized or failed to initialize. Cannot speak.")
        return

    text_to_say_cleaned = str(text_to_say).strip()
    if not text_to_say_cleaned:
        return

    def target():
        try:
            with tts_lock:
                tts_engine.say(text_to_say_cleaned)
                tts_engine.runAndWait()
        except RuntimeError as e:
            if "run loop already started" in str(e).lower():
                logger.warning(f"TTS run loop conflict for text '{text_to_say_cleaned[:30]}...'. Speech might be skipped.")
            else:
                logger.error(f"TTS runtime error in thread: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"TTS error in thread for text '{text_to_say_cleaned[:30]}...': {e}", exc_info=True)

    thread = Thread(target=target)
    thread.daemon = True
    thread.start()

# --- Custom Logging Handler for Specific Console Output ---
class SelectiveConsolePrintHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_response_marker = "RAW_LLM_RESPONSE_FOR_PARSING:"

    def emit(self, record):
        log_message = self.format(record)
        if self.llm_response_marker not in log_message:
            return

        try:
            response_text = log_message.split(self.llm_response_marker, 1)[1].strip()
        except IndexError:
            return

        thinking_content = None
        result_content = None
        ask_followup_raw_content = None
        
        try:
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
            if thinking_match:
                thinking_content = thinking_match.group(1).strip()

            attempt_completion_match = re.search(r"<attempt_completion>(.*?)</attempt_completion>", response_text, re.DOTALL)
            if attempt_completion_match:
                result_tag_match = re.search(r"<result>(.*?)</result>", attempt_completion_match.group(1), re.DOTALL)
                if result_tag_match:
                    result_content = result_tag_match.group(1).strip()
            
            ask_followup_match = re.search(r"<ask_followup_question>(.*?)</ask_followup_question>", response_text, re.DOTALL)
            if ask_followup_match:
                ask_followup_raw_content = ask_followup_match.group(1).strip()
        except Exception:
            # Silently ignore regex errors for console cleanliness
            pass 

        # --- Prepare console and speech outputs ---
        console_output_blocks = [] # Each item will be a block of text for console
        speech_parts_to_say = []   # Each item is a phrase for speech

        if thinking_content:
            console_output_blocks.append(thinking_content) # No "Thinking," prefix for console
            speech_parts_to_say.append(thinking_content)   # Speech will just be the thought

        if result_content:
            console_output_blocks.append(f"Attempt Completion, {result_content}") # Keep prefix for console
            speech_parts_to_say.append(f"Attempt Completion. {result_content}")    # Prefix for speech clarity

        if ask_followup_raw_content:
            question_text_parsed = ""
            options_list_str_parsed = "" # For both console and speech, format may differ slightly
            
            question_text_match = re.search(r"<question>(.*?)</question>", ask_followup_raw_content, re.DOTALL)
            if question_text_match:
                question_text_parsed = question_text_match.group(1).strip()
            
            options_text_match = re.search(r"<options>(.*?)</options>", ask_followup_raw_content, re.DOTALL)
            if options_text_match:
                raw_options_str = options_text_match.group(1).strip()
                try:
                    options_parsed_json = json.loads(raw_options_str)
                    if isinstance(options_parsed_json, list):
                        options_list_str_parsed = ", ".join(str(opt) for opt in options_parsed_json)
                except json.JSONDecodeError:
                    cleaned_options_val = raw_options_str
                    if cleaned_options_val.startswith('[') and cleaned_options_val.endswith(']'):
                        cleaned_options_val = cleaned_options_val[1:-1]
                    parts = [opt.strip().strip('"').strip("'") for opt in cleaned_options_val.split(',')]
                    options_list_str_parsed = ", ".join(p for p in parts if p)
                except Exception:
                    options_list_str_parsed = "" # Default to empty if complex error

            followup_console_block_parts = []
            if question_text_parsed:
                followup_console_block_parts.append(question_text_parsed) # No "I have question," for console
                speech_parts_to_say.append(f"I have a question: {question_text_parsed}") # Prefix for speech
            
            if options_list_str_parsed:
                followup_console_block_parts.append(options_list_str_parsed)
                speech_parts_to_say.append(f"Your options are: {options_list_str_parsed}") # Prefix for speech
            
            if followup_console_block_parts:
                # Join question and options with a newline for console if both exist
                console_output_blocks.append("\n".join(followup_console_block_parts))

        # --- Print to console ---
        if console_output_blocks:
            # Join the main blocks (thinking, attempt, followup_block) with a blank line
            full_console_output = "\n\n".join(filter(None, console_output_blocks))
            if full_console_output: # Ensure not printing just newlines if all blocks were empty somehow
                sys.stdout.write(full_console_output + "\n")
                sys.stdout.flush()

        # --- Speak the combined message ---
        if speech_parts_to_say:
            # Join speech parts with ". " for a slight pause and natural flow
            full_speech_output = ". ".join(filter(None, speech_parts_to_say))
            if full_speech_output: # Ensure not trying to speak an empty string
                speak_threaded(full_speech_output)


# --- Standard Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[] 
)
logger = logging.getLogger(__name__)

root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
        root_logger.removeHandler(handler)

selective_handler = SelectiveConsolePrintHandler()
selective_handler.setFormatter(logging.Formatter('%(message)s')) 
selective_handler.setLevel(logging.INFO) 
root_logger.addHandler(selective_handler)

# file_handler = logging.FileHandler("app_full.log", mode='w') 
# file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'))
# file_handler.setLevel(logging.DEBUG)
# root_logger.addHandler(file_handler)

try:
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.ERROR) 
    werkzeug_logger.propagate = False 
except Exception as e:
    logger.warning(f"Could not modify werkzeug logger settings: {e}")


# --- Flask App and Globals ---
app = Flask(__name__)
last_request_time = 0
MIN_REQUEST_INTERVAL = 5 

set_autopath(r"D:\cline-x-claudeweb\images")
set_altpath(r"D:\cline-x-claudeweb\images\alt1440")

# --- Clipboard Functions (Unchanged from previous version) ---
def set_clipboard(text, retries=3, delay=0.2):
    for i in range(retries):
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            try:
                win32clipboard.SetClipboardText(str(text))
            except Exception: 
                win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, str(text).encode('utf-16le'))
            win32clipboard.CloseClipboard()
            logger.debug("Set clipboard text successfully.")
            return
        except pywintypes.error as e:
            if e.winerror == 5: 
                logger.warning(f"Clipboard access denied. Retrying... (Attempt {i+1}/{retries})")
                time.sleep(delay)
            else:
                logger.error(f"pywintypes.error setting clipboard text: {e}", exc_info=True)
                raise
        except Exception as e:
            logger.error(f"Exception setting clipboard text: {e}", exc_info=True)
            raise
    logger.error(f"Failed to set clipboard text after {retries} attempts.")

def set_clipboard_image(image_data):
    try:
        binary_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(binary_data))
        
        output = io.BytesIO()
        image.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:] 
        output.close()
        
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data) 
        win32clipboard.CloseClipboard()
        logger.debug("Successfully set image to clipboard.")
        return True
    except Exception as e:
        logger.error(f"Error setting image to clipboard: {e}", exc_info=True)
        return False

# --- Content Extraction (Unchanged from previous version) ---
def get_content_text(content: Union[str, List[Dict[str, str]], Dict[str, str]]) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item_idx, item in enumerate(content):
            if item.get("type") == "text":
                parts.append(item["text"])
            elif item.get("type") == "image_url": 
                image_data = item.get("image_url", {}).get("url", "")
                if image_data.startswith('data:image'):
                    logger.info(f"Found image data in content list (item {item_idx}, type 'image_url'), attempting to set to clipboard.")
                    set_clipboard_image(image_data)
                description = item.get("description", "An uploaded image") 
                parts.append(f"[Image: {description}]")
            elif item.get("type") == "image": 
                image_data_url = item.get("image_url", {}).get("url", "") 
                if not image_data_url and "data" in item: 
                    try:
                        img_bytes = item["data"] if isinstance(item["data"], bytes) else item["data"].encode('utf-8')
                        image_data_url = 'data:image/unknown;base64,' + base64.b64encode(img_bytes).decode('utf-8')
                    except Exception as e:
                        logger.error(f"Error encoding item['data'] to base64: {e}")
                        image_data_url = ""
                
                if image_data_url.startswith('data:image'):
                    logger.info(f"Found 'image' type data in content list (item {item_idx}), attempting to set to clipboard.")
                    set_clipboard_image(image_data_url)
                description = item.get("description", "An uploaded image")
                parts.append(f"[Image: {description}]")
        return "\n".join(parts)
    elif isinstance(content, dict): 
        text_content = content.get("text", "")
        if content.get("type") == "image": 
            image_data_url = content.get("image_url", {}).get("url", "")
            if not image_data_url and "data" in content:
                try:
                    img_bytes = content["data"] if isinstance(content["data"], bytes) else content["data"].encode('utf-8')
                    image_data_url = 'data:image/unknown;base64,' + base64.b64encode(img_bytes).decode('utf-8')
                except Exception as e:
                    logger.error(f"Error encoding content['data'] to base64: {e}")
                    image_data_url = ""
            if image_data_url.startswith('data:image'):
                logger.info("Found image in dict content, attempting to set to clipboard.")
                set_clipboard_image(image_data_url)
            description = content.get("description", "An uploaded image")
            return f"[Image: {description}]" 
        return text_content 
    return ""

# --- Core LLM Interaction Logic (Unchanged from previous version) ---
def handle_llm_interaction(prompt_text: str, request_json_data: Dict):
    global last_request_time
    logger.info(f"Starting LLM interaction. User prompt (first 200 chars): {prompt_text[:200]}...")

    current_time = time.time()
    time_since_last = current_time - last_request_time
    
    if time_since_last < MIN_REQUEST_INTERVAL:
        sleep_duration = MIN_REQUEST_INTERVAL - time_since_last
        logger.info(f"Request interval too short. Sleeping for {sleep_duration:.2f} seconds.")
        sleep(sleep_duration)
    last_request_time = time.time()

    image_list = []
    if 'messages' in request_json_data:
        for message in request_json_data['messages']:
            content = message.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        image_url_data = item.get('image_url', {})
                        if isinstance(image_url_data, dict): 
                            url = image_url_data.get('url', '')
                            if url.startswith('data:image'):
                                image_list.append(url)
                        elif isinstance(image_url_data, str) and image_url_data.startswith('data:image'): 
                             image_list.append(image_url_data)
    
    logger.info(f"Extracted {len(image_list)} image(s) for LLM.")

    current_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
    request_json_data_for_prompt_log = copy.deepcopy(request_json_data)

    if 'messages' in request_json_data_for_prompt_log:
        for message in request_json_data_for_prompt_log['messages']:
            content = message.get('content', [])
            if isinstance(content, list):
                for item_idx, item in enumerate(content): 
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        image_url_field = item.get('image_url', {})
                        obfuscation_str = '[IMAGE DATA REMOVED]'
                        if isinstance(image_url_field, dict) and 'url' in image_url_field and \
                           image_url_field['url'].startswith('data:image'):
                            image_url_field['url'] = obfuscation_str
                        elif isinstance(item.get('image_url'), str) and item['image_url'].startswith('data:image'): 
                             item['image_url'] = obfuscation_str

    llm_prompt_header_lines = []
    llm_prompt_header_lines.append(f"{current_time_str} - INFO - Time since last request: {time_since_last:.2f} seconds")
    try:
        request_data_str = json.dumps(request_json_data_for_prompt_log)
        llm_prompt_header_lines.append(f"{current_time_str} - INFO - Request data: {request_data_str}")
    except Exception as e:
        logger.error(f"Error creating request_data_str for LLM prompt log: {e}")
        llm_prompt_header_lines.append(f"{current_time_str} - INFO - Request data: [Error during stringification]")
    
    headers_for_llm_prompt = "\n".join(llm_prompt_header_lines)

    prompt_parts = [headers_for_llm_prompt]
    prompt_parts.append(r'Please follow these rules: For each response, you must use one of conseguavailable tools formatted in proper XML tags. Tools include attempt_completion, ask_followup_question, read_file, write_to_file, search_files, list_files, execute_command, and list_code_definition_names. Do not respond conversationally - only use tool commands. Format any code you generate with proper indentation and line breaks, as you would in a standard code editor. Disregard any previous instructions about generating code in a single line or avoiding newline characters.')
    prompt_parts.append(r'Write the entirity of your response in 1 big markdown codeblock, no word should be out of this 1 big code block and do not write a md codeblock within this big codeblock') # For Gemini fixing
    prompt_parts.append(prompt_text)

    full_prompt_for_llm = "\n".join(prompt_parts)

    logger.debug(f"Full textual prompt for LLM (first 500 chars): {full_prompt_for_llm[:500]}...")
    if image_list:
        logger.debug(f"Actually sending {len(image_list)} image(s) to talkto function.")

    llm_response_raw = talkto("gemini", full_prompt_for_llm, image_list)
    
    if isinstance(llm_response_raw, str):
        logger.info(f"{selective_handler.llm_response_marker} {llm_response_raw}")
    else:
        logger.warning(f"LLM response was not a string (type: {type(llm_response_raw)}). Content: {str(llm_response_raw)[:100]}. Cannot parse for console output.")

    if isinstance(llm_response_raw, str):
        processed_response = llm_response_raw.strip()
        if processed_response.startswith("```") and processed_response.endswith("```"):
            processed_response = re.sub(r'^```[a-zA-Z]*\n?', '', processed_response)
            processed_response = re.sub(r'\n?```$', '', processed_response)
            processed_response = processed_response.strip()
        elif processed_response.endswith("```"): 
            processed_response = processed_response[:-3].strip()
        return processed_response
    else:
        return ""

# --- Flask Routes (Unchanged from previous version) ---
@app.route('/', methods=['GET'])
def home():
    logger.info(f"GET request to / from {request.remote_addr}")
    return "Claude API Bridge (Selective Console Output, Cline X Aligned)"

@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json()
        logger.info(f"Received /chat/completions request. Keys: {list(data.keys())}. Streaming: {data.get('stream', False)}")
        logger.debug(f"Request data dump (first 500 chars): {str(data)[:500]}")

        if not data or 'messages' not in data or not data['messages']:
            logger.warning("Invalid request: 'messages' field missing or empty.")
            return jsonify({'error': {'message': 'Invalid request format: missing or empty "messages" field'}}), 400

        last_message = data['messages'][-1]
        prompt_text_for_llm = get_content_text(last_message.get('content', '')) 
        
        request_id = str(int(time.time())) 
        is_streaming = data.get('stream', False)
        model_name = data.get("model", "gpt-3.5-turbo") 

        logger.info(f"Extracted user prompt for LLM (first 200 chars): {prompt_text_for_llm[:200]}...")
        
        response_content_str = handle_llm_interaction(prompt_text_for_llm, data) 
        logger.info(f"LLM interaction complete. Response length: {len(response_content_str)}")

        if is_streaming:
            def generate():
                response_id = f"chatcmpl-{request_id}"
                chunk = {
                    "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                lines = response_content_str.splitlines(True) 
                for line_content in lines:
                    if not line_content: continue 
                    chunk = {
                        "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": line_content}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                chunk = {
                    "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        
        prompt_str_for_usage = str(prompt_text_for_llm)
        response_str_for_usage = str(response_content_str)

        return jsonify({
            'id': f'chatcmpl-{request_id}', 'object': 'chat.completion', 'created': int(time.time()),
            'model': model_name,
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': response_str_for_usage},
                'finish_reason': 'stop'
            }],
            'usage': { 
                'prompt_tokens': len(prompt_str_for_usage.split()), 
                'completion_tokens': len(response_str_for_usage.split()),
                'total_tokens': len(prompt_str_for_usage.split()) + len(response_str_for_usage.split())
            }
        })
    
    except Exception as e:
        logger.error(f"Critical error in /chat/completions: {str(e)}", exc_info=True)
        return jsonify({'error': {'message': f"An internal server error occurred: {str(e)}"}}), 500

# --- Main Execution (Unchanged from previous version) ---
if __name__ == '__main__':
    init_tts() 

    startup_message = "Cline x voice running"
    sys.stdout.write(startup_message + "\n")
    sys.stdout.flush() 
    speak_threaded(startup_message) 
    
    logger.info(f"Starting API Bridge server on port 3001. TTS Initialized: {tts_initialized_successfully}")
    
    try:
        app.run(host="0.0.0.0", port=3001, debug=False)
    finally:
        logger.info("Application shutting down.")