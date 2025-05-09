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
import sys # For custom print handler

# --- Configuration Reading ---
def read_config(filename="config.txt"):
    config_values = {} # Renamed to avoid conflict with module 'config' if it exists
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): # Skip empty lines and comments
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    config_values[key.strip()] = str(value.strip().strip('"').strip("'"))
    except FileNotFoundError:
        # Log this, but proceed with defaults
        # This will be logged by the root logger setup later
        # For now, a print statement if logger is not yet configured at this point in execution
        print(f"Warning: {filename} not found. Using default configurations.", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Error reading {filename}: {e}. Using default configurations.", file=sys.stderr)
    return config_values

config_data = read_config()
# Default to False if not specified or if config.txt is missing/malformed
autorun = config_data.get('autorun', 'False').lower() == 'true'
usefirefox = config_data.get('usefirefox', 'False').lower() == 'true'


# --- Custom Logging Handler for Specific Console Output ---
class SelectiveConsolePrintHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_response_marker = "RAW_LLM_RESPONSE_FOR_PARSING:"

    def emit(self, record):
        log_message = self.format(record)
        if self.llm_response_marker in log_message:
            try:
                response_text = log_message.split(self.llm_response_marker, 1)[1].strip()
            except IndexError:
                return

            thinking_content = None
            result_content = None
            ask_followup_content = None
            printed_anything = False

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
                    ask_followup_content = ask_followup_match.group(1).strip()

            except Exception:
                pass # Silently ignore regex errors for console cleanliness

            if thinking_content:
                sys.stdout.write("Thinking:\n")
                sys.stdout.write(thinking_content + "\n")
                printed_anything = True

            if result_content:
                if printed_anything:
                    sys.stdout.write("\n")
                sys.stdout.write(f"Attempt Completion: {result_content}\n")
                printed_anything = True
            
            if ask_followup_content:
                if printed_anything:
                    sys.stdout.write("\n")
                sys.stdout.write(f"Ask Followup Question: {ask_followup_content}\n")

            if printed_anything:
                sys.stdout.flush()

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

# --- Optionally, add a FileHandler ---
# file_handler = logging.FileHandler("app_full.log", mode='w') # mode='w' to overwrite on startup
# file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'))
# file_handler.setLevel(logging.DEBUG)
# root_logger.addHandler(file_handler)

try:
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.ERROR)
except Exception as e:
    logger.warning(f"Could not modify werkzeug logger settings: {e}")


# --- Flask App and Globals ---
app = Flask(__name__)
last_request_time = 0
MIN_REQUEST_INTERVAL = 5

set_autopath(r"D:\cline-x-claudeweb\images")
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
                    set_clipboard_image(image_data) # Assuming set_clipboard_image can handle the "url" field directly if it's a data URI
                description = item.get("description", "An uploaded image")
                parts.append(f"[Image: {description}]")
            elif item.get("type") == "image":
                image_data_url = item.get("image_url", {}).get("url", "")
                if not image_data_url and "data" in item:
                    try: # Ensure item["data"] is bytes before b64encode
                        img_bytes = item["data"] if isinstance(item["data"], bytes) else item["data"].encode('utf-8')
                        image_data_url = 'data:image/unknown;base64,' + base64.b64encode(img_bytes).decode('utf-8')
                    except Exception as e:
                        logger.error(f"Error encoding item['data'] to base64: {e}")
                        image_data_url = "" # Avoid further errors

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
            return f"[Image: {description}]" # Return only image placeholder if it's an image type dict
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
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        image_url_field = item.get('image_url', {})
                        obfuscation_str = '[IMAGE DATA REMOVED]' # Match old script's obfuscation string
                        if isinstance(image_url_field, dict) and 'url' in image_url_field and \
                           image_url_field['url'].startswith('data:image'):
                            image_url_field['url'] = obfuscation_str
                        elif isinstance(item.get('image_url'), str) and item['image_url'].startswith('data:image'):
                             item['image_url'] = obfuscation_str


    # Construct the "header" part of the prompt for the LLM
    # This aims to match the old script's format for these lines.
    llm_prompt_header_lines = []
    llm_prompt_header_lines.append(f"{current_time_str} - INFO - Time since last request: {time_since_last:.2f} seconds")
    try:
        request_data_str = json.dumps(request_json_data_for_prompt_log)
        # Removed truncation to match old script's behavior of sending the full (obfuscated) request JSON.
        llm_prompt_header_lines.append(f"{current_time_str} - INFO - Request data: {request_data_str}")
    except Exception as e:
        logger.error(f"Error creating request_data_str for LLM prompt log: {e}")
        llm_prompt_header_lines.append(f"{current_time_str} - INFO - Request data: [Error during stringification]")
    
    headers_for_llm_prompt = "\n".join(llm_prompt_header_lines)

    # Assemble the full textual prompt
    prompt_parts = [headers_for_llm_prompt]

    if autorun: # Global `autorun` boolean from config.txt
        prompt_parts.append(r'You are set to autorun mode which means you cant use attempt completion or ask follow up questions, you can only write code and use terminal, so if you need something like a database or something, work it out yourself. Dont run anything in terminal that asks for input after you have run the command.')
    
    prompt_parts.append(r'Please follow these rules: For each response, you must use one of conseguavailable tools formatted in proper XML tags. Tools include attempt_completion, ask_followup_question, read_file, write_to_file, search_files, list_files, execute_command, and list_code_definition_names. Do not respond conversationally - only use tool commands. Format any code you generate with proper indentation and line breaks, as you would in a standard code editor. Disregard any previous instructions about generating code in a single line or avoiding newline characters.')
    prompt_parts.append(r'Write the entirity of your response in 1 big markdown codeblock, no word should be out of this 1 big code block and do not write a md codeblock within this big codeblock') # For Gemini fixing
    prompt_parts.append(prompt_text)

    full_prompt_for_llm = "\n".join(prompt_parts) # Use single newline to join parts, as in old script.

    logger.debug(f"Full textual prompt for LLM (first 500 chars): {full_prompt_for_llm[:500]}...")
    if image_list:
        logger.debug(f"Actually sending {len(image_list)} image(s) to talkto function.")

    llm_response_raw = talkto("gemini", full_prompt_for_llm, image_list)
    
    if isinstance(llm_response_raw, str):
        logger.info(f"{selective_handler.llm_response_marker} {llm_response_raw}")
    else:
        logger.warning(f"LLM response was not a string (type: {type(llm_response_raw)}). Content: {str(llm_response_raw)[:100]}. Cannot parse for console output.")

    if isinstance(llm_response_raw, str):
        # Match old script's logic more closely for stripping '```', then general strip.
        # The old script did `[:-3]` unconditionally if it was always expected.
        # The new script was safer, but if Cline X expects ` ``` ` to be there and be stripped:
        if llm_response_raw.endswith("```"):
            processed_response = llm_response_raw[:-3].strip()
        else:
            # If the old script *always* removed the last 3 chars, uncomment next line.
            # This would be risky if response doesn't end with ```.
            # processed_response = llm_response_raw[:-3].strip() if len(llm_response_raw) > 3 else llm_response_raw.strip()
            processed_response = llm_response_raw.strip() # Current safer approach
        return processed_response
    else:
        return ""

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
        
        request_id = str(int(time.time()))
        is_streaming = data.get('stream', False)
        model_name = data.get("model", "gpt-3.5-turbo") # Use requested model or default

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

                # Using splitlines(True) to keep newlines, which is generally good for streaming.
                # The old script did `line + "\n"`. This should be equivalent or better.
                lines = response_content_str.splitlines(True) 
                for line_content in lines:
                    if not line_content: continue
                    chunk = {
                        "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": line_content}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    # sleep(0.01) # Optional small delay

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

# --- Main Execution ---
if __name__ == '__main__':
    logger.info(f"Attempting to read configuration from config.txt...")
    logger.info(f"Configuration loaded: Autorun: {autorun}, UseFirefox: {usefirefox}")
    logger.info(f"Starting API Bridge server on port 3001.")
    
    app.run(host="0.0.0.0", port=3001, debug=False)