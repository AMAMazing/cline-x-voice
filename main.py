from flask import Flask, jsonify, request, Response
import win32clipboard
import time
import pywintypes
from time import sleep
from optimisewait import set_autopath, set_altpath
import logging
import json
from typing import Union, List, Dict, Optional
import base64
import io
from PIL import Image
import re
from talktollm import talkto # Assuming this is correctly implemented elsewhere
import copy
import sys

# --- TTS Imports and Setup ---
import pyttsx3
from threading import Thread, Lock

tts_engine = None
tts_initialized_successfully = False
tts_lock = Lock()
init_attempts = 0 # Counter for init attempts

# --- Standard Logging Setup (configured early) ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[]
)
logger = logging.getLogger(__name__) # Main logger for the application

root_logger_obj = logging.getLogger() # Get the root logger

# Remove any default console handlers that might have been added by basicConfig or Flask
for handler in root_logger_obj.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
        root_logger_obj.removeHandler(handler)

# FileHandler for full debug logs (configured before TTS init uses logger)
file_handler = logging.FileHandler("app_full.log", mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'))
file_handler.setLevel(logging.DEBUG)
root_logger_obj.addHandler(file_handler)
# --- End FileHandler setup ---

def init_tts():
    global tts_engine, tts_initialized_successfully, init_attempts
    init_attempts += 1
    logger.info(f"Attempting to initialize TTS engine (Attempt: {init_attempts})...")

    if tts_engine is not None and tts_initialized_successfully:
        logger.debug(f"TTS init_tts (Attempt: {init_attempts}): Engine already initialized and flagged successful.")
        return
    if tts_engine is not None and not tts_initialized_successfully: # Engine object exists but failed init
        logger.warning(f"TTS init_tts (Attempt: {init_attempts}): Engine object exists ({id(tts_engine)}) but was not flagged successful. Re-attempting init.")
        # Potentially try to clean up old engine object if pyttsx3 allows/requires it
        # For now, just overwrite it.
    
    try:
        # Explicitly set SAPI5 driver if on Windows, for more control
        # On other OS, pyttsx3 will pick default.
        if sys.platform == "win32":
            logger.debug("TTS init_tts: Attempting to init with SAPI5 driver explicitly.")
            tts_engine = pyttsx3.init(driverName='sapi5')
        else:
            tts_engine = pyttsx3.init()
            
        if tts_engine:
            logger.info(f"pyttsx3.init() was successful (Attempt: {init_attempts}), engine object created: {id(tts_engine)}")
            
            try:
                voices = tts_engine.getProperty('voices')
                if voices:
                    logger.debug(f"Available TTS voices: {[v.name for v in voices]}")
                    # Example: Set a specific voice if needed (e.g., voices[0].id)
                else:
                    logger.warning("No TTS voices found by pyttsx3.")
            except Exception as e_voices:
                logger.warning(f"Could not get/set voices during TTS init: {e_voices}")

            test_phrase = "TTS engine initialized and speaking test phrase."
            logger.info(f"Attempting to speak test phrase directly: '{test_phrase}' (Attempt: {init_attempts})")
            tts_engine.say(test_phrase)
            tts_engine.runAndWait() 
            logger.info(f"Direct test phrase spoken (Attempt: {init_attempts}).")
            
            tts_initialized_successfully = True
            logger.info(f"TTS engine has been marked as initialized successfully (Attempt: {init_attempts}). Engine ID: {id(tts_engine)}")
        else:
            logger.error(f"pyttsx3.init() returned None. TTS initialization failed (Attempt: {init_attempts}).")
            tts_initialized_successfully = False
            tts_engine = None

    except Exception as e:
        logger.error(f"Exception during TTS engine initialization (Attempt: {init_attempts}): {e}", exc_info=True)
        tts_engine = None
        tts_initialized_successfully = False

def speak_threaded(text_to_say: str):
    logger.debug(f"speak_threaded called with text (first 100 chars): '{str(text_to_say)[:100]}'")
    if not tts_initialized_successfully:
        logger.warning("TTS speak_threaded: tts_initialized_successfully is False. Cannot speak.")
        return

    if tts_engine is None:
        logger.warning("TTS speak_threaded: tts_engine is None. Cannot speak.")
        return

    text_to_say_cleaned = str(text_to_say).strip()
    if not text_to_say_cleaned:
        logger.debug("speak_threaded: Cleaned text is empty. Not speaking.")
        return

    logger.debug(f"speak_threaded: Proceeding to create thread for speaking: '{text_to_say_cleaned[:100]}'")

    def target():
        try:
            with tts_lock:
                logger.info(f"TTS Thread: Acquired lock for '{text_to_say_cleaned[:30]}...' Current Engine ID: {id(tts_engine) if tts_engine else 'None'}")
                
                if tts_engine is None or not tts_initialized_successfully: # Re-check inside lock
                    logger.error(f"TTS Thread: Engine became None or uninitialized before speaking '{text_to_say_cleaned[:30]}...'")
                    return

                try:
                    current_voice_obj = tts_engine.getProperty('voice') # This returns voice ID string
                    current_rate = tts_engine.getProperty('rate')
                    current_volume = tts_engine.getProperty('volume')
                    logger.debug(f"TTS Thread: Engine props before say(): VoiceID='{current_voice_obj}', Rate={current_rate}, Vol={current_volume}. Engine ID: {id(tts_engine)}")
                except Exception as e_prop:
                    logger.error(f"TTS Thread: Failed to get engine properties for '{text_to_say_cleaned[:30]}...': {e_prop}", exc_info=True)
                    # If props fail, engine is likely unusable. Avoid trying to use it.
                    # global tts_initialized_successfully # To modify it
                    # tts_initialized_successfully = False
                    # logger.error("TTS Thread: Marked TTS as uninitialized due to property access error.")
                    return # Don't attempt to speak

                logger.info(f"TTS Thread: Attempting to say: '{text_to_say_cleaned[:100]}...' with engine {id(tts_engine)}")
                tts_engine.say(text_to_say_cleaned)
                logger.debug(f"TTS Thread: say() called for '{text_to_say_cleaned[:30]}...'. Calling runAndWait()...")
                tts_engine.runAndWait() 
                logger.info(f"TTS Thread: Finished speaking (runAndWait completed for): '{text_to_say_cleaned[:100]}...'")
        
        except RuntimeError as e: 
            if "run loop already started" in str(e).lower() or "event loop" in str(e).lower():
                logger.warning(f"TTS run loop conflict for text '{text_to_say_cleaned[:30]}...'. Speech might be skipped. Error: {e}")
            else: 
                logger.error(f"TTS Thread: Unexpected RuntimeError for '{text_to_say_cleaned[:30]}...': {e}", exc_info=True)
        except Exception as e: 
            logger.error(f"TTS Thread: Generic error during say/runAndWait for '{text_to_say_cleaned[:30]}...': {e}", exc_info=True)

    thread = Thread(target=target)
    thread.daemon = True 
    thread.start()
    logger.debug(f"speak_threaded: Speech thread started for: '{text_to_say_cleaned[:100]}'")


# --- Custom Logging Handler for Specific Console Output ---
class SelectiveConsolePrintHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_response_marker = "RAW_LLM_RESPONSE_FOR_PARSING:"

    def emit(self, record):
        log_message = self.format(record)
        if self.llm_response_marker not in log_message:
            return

        logger.debug(f"SelectiveConsolePrintHandler received LLM log: {log_message[:200]}...")

        try:
            response_text = log_message.split(self.llm_response_marker, 1)[1].strip()
        except IndexError:
            logger.warning("SelectiveConsolePrintHandler: Could not split LLM response marker.")
            return

        thinking_content = None
        result_content = None
        ask_followup_raw_content = None
        
        try:
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
            if thinking_match:
                thinking_content = thinking_match.group(1).strip()
                logger.debug(f"Parsed thinking_content: {thinking_content[:100]}...")

            attempt_completion_match = re.search(r"<attempt_completion>(.*?)</attempt_completion>", response_text, re.DOTALL)
            if attempt_completion_match:
                result_tag_match = re.search(r"<result>(.*?)</result>", attempt_completion_match.group(1), re.DOTALL)
                if result_tag_match:
                    result_content = result_tag_match.group(1).strip()
                    logger.debug(f"Parsed result_content: {result_content[:100]}...")
            
            ask_followup_match = re.search(r"<ask_followup_question>(.*?)</ask_followup_question>", response_text, re.DOTALL)
            if ask_followup_match:
                ask_followup_raw_content = ask_followup_match.group(1).strip()
                logger.debug(f"Parsed ask_followup_raw_content: {ask_followup_raw_content[:100]}...")
        except Exception as e:
            logger.error(f"SelectiveConsolePrintHandler: Error during regex parsing: {e}", exc_info=True)
            pass 

        console_output_blocks = [] 
        speech_parts_to_say = []   

        if thinking_content:
            console_output_blocks.append(thinking_content) 
            speech_parts_to_say.append(thinking_content)   

        if result_content:
            console_output_blocks.append(f"Attempt Completion, {result_content}") 
            speech_parts_to_say.append(f"Attempt Completion. {result_content}")    

        if ask_followup_raw_content:
            question_text_parsed = ""
            options_list_str_parsed = ""
            
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
                    options_list_str_parsed = ""

            followup_console_block_parts = []
            if question_text_parsed:
                followup_console_block_parts.append(question_text_parsed) 
                speech_parts_to_say.append(f"I have a question: {question_text_parsed}") 
            
            if options_list_str_parsed:
                followup_console_block_parts.append(options_list_str_parsed)
                speech_parts_to_say.append(f"Your options are: {options_list_str_parsed}") 
            
            if followup_console_block_parts:
                console_output_blocks.append("\n".join(followup_console_block_parts))

        if console_output_blocks:
            full_console_output = "\n\n".join(filter(None, console_output_blocks))
            if full_console_output: 
                sys.stdout.write(full_console_output + "\n")
                sys.stdout.flush()
                logger.debug(f"SelectiveConsolePrintHandler: Wrote to console: {full_console_output[:100]}...")

        if speech_parts_to_say:
            full_speech_output = ". ".join(filter(None, speech_parts_to_say)).strip()
            logger.debug(f"SelectiveConsolePrintHandler: Prepared for speech: '{full_speech_output[:100]}...'")
            if full_speech_output:
                speak_threaded(full_speech_output)
            else:
                logger.debug("SelectiveConsolePrintHandler: full_speech_output is empty, not calling speak_threaded.")
        else:
            logger.debug("SelectiveConsolePrintHandler: speech_parts_to_say is empty.")

# Add SelectiveConsolePrintHandler to root logger AFTER basicConfig and FileHandler
selective_handler = SelectiveConsolePrintHandler()
selective_handler.setFormatter(logging.Formatter('%(message)s')) 
selective_handler.setLevel(logging.INFO) 
root_logger_obj.addHandler(selective_handler)

# Suppress Werkzeug's default logging
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

set_autopath(r"D:\cline-x-claudeweb\images") # Ensure these paths are correct for your system
set_altpath(r"D:\cline-x-claudeweb\images\alt1440")

# --- Clipboard Functions ---
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

# --- Content Extraction ---
def get_content_text(content: Union[str, List[Dict[str, str]], Dict[str, str]]) -> str:
    if isinstance(content, str): return content
    elif isinstance(content, list):
        parts = []
        for item_idx, item in enumerate(content):
            if item.get("type") == "text": parts.append(item["text"])
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
                    except Exception as e: logger.error(f"Error encoding item['data'] to base64: {e}"); image_data_url = ""
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
                except Exception as e: logger.error(f"Error encoding content['data'] to base64: {e}"); image_data_url = ""
            if image_data_url.startswith('data:image'):
                logger.info("Found image in dict content, attempting to set to clipboard.")
                set_clipboard_image(image_data_url)
            description = content.get("description", "An uploaded image")
            return f"[Image: {description}]" 
        return text_content 
    return ""

# --- Core LLM Interaction Logic ---
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
                            if url.startswith('data:image'): image_list.append(url)
                        elif isinstance(image_url_data, str) and image_url_data.startswith('data:image'): image_list.append(image_url_data)
    logger.info(f"Extracted {len(image_list)} image(s) for LLM.")
    current_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
    request_json_data_for_prompt_log = copy.deepcopy(request_json_data)
    if 'messages' in request_json_data_for_prompt_log:
        for message in request_json_data_for_prompt_log['messages']:
            content = message.get('content', [])
            if isinstance(content, list):
                for item_idx, item in enumerate(content): 
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        image_url_field = item.get('image_url', {}); obfuscation_str = '[IMAGE DATA REMOVED]'
                        if isinstance(image_url_field, dict) and 'url' in image_url_field and image_url_field['url'].startswith('data:image'): image_url_field['url'] = obfuscation_str
                        elif isinstance(item.get('image_url'), str) and item['image_url'].startswith('data:image'): item['image_url'] = obfuscation_str
    llm_prompt_header_lines = [f"{current_time_str} - INFO - Time since last request: {time_since_last:.2f} seconds"]
    try:
        request_data_str = json.dumps(request_json_data_for_prompt_log)
        llm_prompt_header_lines.append(f"{current_time_str} - INFO - Request data: {request_data_str}")
    except Exception as e:
        logger.error(f"Error creating request_data_str for LLM prompt log: {e}")
        llm_prompt_header_lines.append(f"{current_time_str} - INFO - Request data: [Error during stringification]")
    headers_for_llm_prompt = "\n".join(llm_prompt_header_lines)
    prompt_parts = [headers_for_llm_prompt]
    prompt_parts.append(r'Please follow these rules: For each response, you must use one of conseguavailable tools formatted in proper XML tags. Tools include attempt_completion, ask_followup_question, read_file, write_to_file, search_files, list_files, execute_command, and list_code_definition_names. Do not respond conversationally - only use tool commands. Format any code you generate with proper indentation and line breaks, as you would in a standard code editor. Disregard any previous instructions about generating code in a single line or avoiding newline characters.')
    prompt_parts.append(r'Write the entirity of your response in 1 big markdown codeblock, no word should be out of this 1 big code block and do not write a md codeblock within this big codeblock')
    prompt_parts.append(prompt_text)
    full_prompt_for_llm = "\n".join(prompt_parts)
    logger.debug(f"Full textual prompt for LLM (first 500 chars): {full_prompt_for_llm[:500]}...")
    if image_list: logger.debug(f"Actually sending {len(image_list)} image(s) to talkto function.")
    
    # Ensure talkto is correctly imported and returns a string
    llm_response_raw = talkto("gemini", full_prompt_for_llm, image_list) 
    
    if isinstance(llm_response_raw, str):
        # This is where the SelectiveConsolePrintHandler gets triggered
        logger.info(f"{selective_handler.llm_response_marker} {llm_response_raw}")
    else:
        logger.warning(f"LLM response was not a string (type: {type(llm_response_raw)}). Content: {str(llm_response_raw)[:100]}. Cannot parse for console output.")
        return "" # Return empty if not a string to prevent downstream errors

    if isinstance(llm_response_raw, str): # Process for returning to client
        processed_response = llm_response_raw.strip()
        if processed_response.startswith("```") and processed_response.endswith("```"):
            processed_response = re.sub(r'^```[a-zA-Z]*\n?', '', processed_response)
            processed_response = re.sub(r'\n?```$', '', processed_response)
            processed_response = processed_response.strip()
        elif processed_response.endswith("```"): processed_response = processed_response[:-3].strip()
        return processed_response
    else: return ""

# --- Flask Routes ---
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
        
        request_id = str(int(time.time())); is_streaming = data.get('stream', False); model_name = data.get("model", "gpt-3.5-turbo") 
        logger.info(f"Extracted user prompt for LLM (first 200 chars): {prompt_text_for_llm[:200]}...")
        
        response_content_str = handle_llm_interaction(prompt_text_for_llm, data) 
        logger.info(f"LLM interaction complete. Response length: {len(response_content_str)}")
        
        if is_streaming:
            def generate():
                response_id = f"chatcmpl-{request_id}"
                chunk = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}; yield f"data: {json.dumps(chunk)}\n\n"
                lines = response_content_str.splitlines(True) 
                for line_content in lines:
                    if not line_content: continue 
                    chunk = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name, "choices": [{"index": 0, "delta": {"content": line_content}, "finish_reason": None}]}; yield f"data: {json.dumps(chunk)}\n\n"
                chunk = {"id": response_id, "object": "chat.completion.chunk", "created": int(time.time()), "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}; yield f"data: {json.dumps(chunk)}\n\n"; yield "data: [DONE]\n\n"
            return Response(generate(), mimetype='text/event-stream')
        
        prompt_str_for_usage = str(prompt_text_for_llm); response_str_for_usage = str(response_content_str)
        return jsonify({'id': f'chatcmpl-{request_id}', 'object': 'chat.completion', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': response_str_for_usage}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': len(prompt_str_for_usage.split()), 'completion_tokens': len(response_str_for_usage.split()), 'total_tokens': len(prompt_str_for_usage.split()) + len(response_str_for_usage.split())}})
    
    except Exception as e:
        logger.error(f"Critical error in /chat/completions: {str(e)}", exc_info=True)
        return jsonify({'error': {'message': f"An internal server error occurred: {str(e)}"}}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # init_tts() must be called after logger is configured
    init_tts() 

    startup_message = "Cline x voice running"
    sys.stdout.write(startup_message + "\n")
    sys.stdout.flush() 
    
    if tts_initialized_successfully:
        logger.info("Attempting to speak startup message...")
        speak_threaded(startup_message) 
    else:
        logger.warning("TTS not initialized successfully, skipping startup message speech.")
    
    logger.info(f"Starting API Bridge server on port 3001. TTS Initialized Flag: {tts_initialized_successfully}. Engine ID: {id(tts_engine) if tts_engine else 'None'}")
    
    try:
        # Use use_reloader=False to prevent Flask from starting two instances of the app in debug mode,
        # which can cause issues with singleton resources like the TTS engine.
        app.run(host="0.0.0.0", port=3001, debug=False, use_reloader=False)
    finally:
        logger.info("Application shutting down.")
        if tts_engine and hasattr(tts_engine, 'stop') and tts_initialized_successfully :
            try:
                logger.info("Attempting to stop TTS engine...")
                # tts_engine.stop() # Often not needed for sapi5 with runAndWait, can sometimes cause issues.
                                  # If issues persist, can try enabling it.
                logger.info("TTS engine stop process (if any) initiated.")
            except Exception as e_stop:
                logger.warning(f"Exception during tts_engine.stop(): {e_stop}")