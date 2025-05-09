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

# Configuration values are now hardcoded or defaulted
autorun = False
usefirefox = False

# --- Custom Logging Handler for Specific Console Output ---
class SelectiveConsolePrintHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This marker helps identify the log message containing the raw LLM response
        self.llm_response_marker = "RAW_LLM_RESPONSE_FOR_PARSING:"

    def emit(self, record):
        # We are looking for a specific log message that contains the raw LLM response
        log_message = self.format(record) # Get the formatted log message as a string
        
        if self.llm_response_marker in log_message:
            # Extract the actual response text after the marker
            try:
                # The actual message payload starts after the marker
                response_text = log_message.split(self.llm_response_marker, 1)[1].strip()
            except IndexError:
                # This should not happen if the marker is correctly prepended
                return 

            thinking_content = None
            result_content = None
            ask_followup_content = None
            printed_anything = False

            try:
                # Extract <thinking>
                thinking_match = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
                if thinking_match:
                    thinking_content = thinking_match.group(1).strip()

                # Extract <result> within <attempt_completion>
                attempt_completion_match = re.search(r"<attempt_completion>(.*?)</attempt_completion>", response_text, re.DOTALL)
                if attempt_completion_match:
                    # Search for <result> only within the content of <attempt_completion>
                    result_tag_match = re.search(r"<result>(.*?)</result>", attempt_completion_match.group(1), re.DOTALL)
                    if result_tag_match:
                        result_content = result_tag_match.group(1).strip()
                
                # Extract <ask_followup_question>
                ask_followup_match = re.search(r"<ask_followup_question>(.*?)</ask_followup_question>", response_text, re.DOTALL)
                if ask_followup_match:
                    ask_followup_content = ask_followup_match.group(1).strip()

            except Exception as e:
                # In a production system, you might want to log this error to a file
                # For now, keeping console clean as per user request
                # print(f"Regex error in SelectiveConsolePrintHandler: {e}", file=sys.stderr)
                pass # Silently ignore regex errors for console cleanliness

            # Print extracted parts to sys.stdout directly
            if thinking_content:
                sys.stdout.write("Thinking:\n")
                sys.stdout.write(thinking_content + "\n")
                printed_anything = True

            if result_content:
                if printed_anything: # Add a blank line if thinking was also printed
                    sys.stdout.write("\n")
                sys.stdout.write(f"Attempt Completion: {result_content}\n")
                printed_anything = True
            
            if ask_followup_content:
                if printed_anything:
                    sys.stdout.write("\n")
                sys.stdout.write(f"Ask Followup Question: {ask_followup_content}\n")
                # printed_anything = True # Not strictly needed to set for the last item

            if printed_anything:
                sys.stdout.flush() # Ensure it's written immediately to the console
        # Else: If the log message doesn't contain the marker, this handler does nothing with it.


# --- Standard Logging Setup ---
# Configure root logger: log everything internally (DEBUG level)
# The format specified here will be used by handlers that don't override it (e.g., FileHandler)
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[] # Start with no handlers; we will add them selectively
)
logger = logging.getLogger(__name__) # Get a logger for this module

# --- Remove any default console handlers that basicConfig might have added ---
# This step ensures that only our custom handler controls what from the application's
# logs goes to the console. Werkzeug logs are handled separately.
root_logger = logging.getLogger() # Get the root logger
for handler in root_logger.handlers[:]: # Iterate over a copy of the handlers list
    if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
        # This condition checks if the handler is a StreamHandler outputting to console
        root_logger.removeHandler(handler)

# --- Add our custom handler for selective console printing ---
selective_handler = SelectiveConsolePrintHandler()
# The formatter for the selective handler only needs the message part for parsing
selective_handler.setFormatter(logging.Formatter('%(message)s')) 
# The custom handler should process INFO level messages (where we'll put the raw LLM response)
selective_handler.setLevel(logging.INFO) 
root_logger.addHandler(selective_handler)

# --- Optionally, add a FileHandler to log EVERYTHING to a file for debugging ---
# file_handler = logging.FileHandler("app_full.log")
# file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'))
# file_handler.setLevel(logging.DEBUG) # Log all debug messages and above to the file
# root_logger.addHandler(file_handler)


# --- Try to silence Werkzeug's (Flask's development server) default request logging ---
try:
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.ERROR) # Show only errors from Werkzeug, not its INFO logs
except Exception as e:
    # Log this failure using our logger, which might go to file if configured
    logger.warning(f"Could not modify werkzeug logger settings: {e}")


# --- Flask App and Globals ---
app = Flask(__name__)
last_request_time = 0
MIN_REQUEST_INTERVAL = 5  # Minimum time between new tab creation

# Paths for optimisewait (if used elsewhere or by talkto indirectly)
set_autopath(r"D:\cline-x-claudeweb\images")
set_altpath(r"D:\cline-x-claudeweb\images\alt1440")

# --- Clipboard Functions (with internal logging) ---
def set_clipboard(text, retries=3, delay=0.2):
    for i in range(retries):
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            try:
                win32clipboard.SetClipboardText(str(text))
            except Exception:
                # Fallback for Unicode characters
                win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, str(text).encode('utf-16le'))
            win32clipboard.CloseClipboard()
            logger.debug("Set clipboard text successfully.") # Internal log
            return  # Success
        except pywintypes.error as e:
            if e.winerror == 5:  # Access is denied
                logger.warning(f"Clipboard access denied. Retrying... (Attempt {i+1}/{retries})") # Internal log
                time.sleep(delay)
            else:
                logger.error(f"pywintypes.error setting clipboard text: {e}", exc_info=True) # Internal log
                raise  # Re-raise other pywintypes errors
        except Exception as e:
            logger.error(f"Exception setting clipboard text: {e}", exc_info=True) # Internal log
            raise  # Re-raise other exceptions
    logger.error(f"Failed to set clipboard text after {retries} attempts.") # Internal log

def set_clipboard_image(image_data):
    """Set image data to clipboard"""
    try:
        # Decode base64 image
        binary_data = base64.b64decode(image_data.split(',')[1])
        
        # Convert to bitmap using PIL
        image = Image.open(io.BytesIO(binary_data))
        
        # Convert to bitmap format
        output = io.BytesIO()
        image.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]  # Remove bitmap header
        output.close()
        
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()
        logger.debug("Successfully set image to clipboard.") # Internal log
        return True
    except Exception as e:
        logger.error(f"Error setting image to clipboard: {e}", exc_info=True) # Internal log
        return False

# --- Content Extraction (with internal logging) ---
def get_content_text(content: Union[str, List[Dict[str, str]], Dict[str, str]]) -> str:
    """Extract text and handle images from different content formats"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for item_idx, item in enumerate(content): # Added index for logging
            if item.get("type") == "text":
                parts.append(item["text"])
            elif item.get("type") == "image_url": # Handles OpenAI image_url format
                image_data = item.get("image_url", {}).get("url", "")
                if image_data.startswith('data:image'):
                    logger.info(f"Found image data in content list (item {item_idx}, type 'image_url'), attempting to set to clipboard.") # Internal log
                    set_clipboard_image(image_data)
                description = item.get("description", "An uploaded image") # Placeholder
                parts.append(f"[Image: {description}]") # Textual representation for the prompt
            elif item.get("type") == "image": # Handling potential other 'image' type (as in original)
                image_data_url = item.get("image_url", {}).get("url", "")
                if not image_data_url and "data" in item:  # For binary image data
                    image_data_url = 'data:image/unknown;base64,' + base64.b64encode(item["data"]).decode('utf-8')
                
                if image_data_url.startswith('data:image'):
                    logger.info(f"Found 'image' type data in content list (item {item_idx}), attempting to set to clipboard.") # Internal log
                    set_clipboard_image(image_data_url)
                description = item.get("description", "An uploaded image")
                parts.append(f"[Image: {description}]")
        return "\n".join(parts)
    elif isinstance(content, dict): # This case might be less common for OpenAI's 'messages' content
        text_content = content.get("text", "")
        if content.get("type") == "image":
            image_data_url = content.get("image_url", {}).get("url", "")
            if not image_data_url and "data" in content:
                 image_data_url = 'data:image/unknown;base64,' + base64.b64encode(content["data"]).decode('utf-8')
            if image_data_url.startswith('data:image'):
                logger.info("Found image in dict content, attempting to set to clipboard.") # Internal log
                set_clipboard_image(image_data_url)
            description = content.get("description", "An uploaded image")
            return f"[Image: {description}]"
        return text_content
    return ""

# --- Core LLM Interaction Logic (with internal logging, and special log for parsing) ---
def handle_llm_interaction(prompt_text: str, request_json_data: Dict): # Renamed prompt to prompt_text for clarity
    global last_request_time
    # This log will go to file if file_handler is active, but not to console due to selective_handler
    logger.info(f"Starting LLM interaction. User prompt (first 200 chars): {prompt_text[:200]}...")

    current_time = time.time()
    time_since_last = current_time - last_request_time
    
    if time_since_last < MIN_REQUEST_INTERVAL:
        sleep_duration = MIN_REQUEST_INTERVAL - time_since_last
        logger.info(f"Request interval too short. Sleeping for {sleep_duration:.2f} seconds.") # Internal log
        sleep(sleep_duration)
    last_request_time = time.time() # Update after potential sleep

    # Extract actual image data for talkto function's image_list argument
    image_list = []
    if 'messages' in request_json_data:
        for message in request_json_data['messages']:
            content = message.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        image_url_data = item.get('image_url', {})
                        if isinstance(image_url_data, dict): # Standard OpenAI format
                            url = image_url_data.get('url', '')
                            if url.startswith('data:image'):
                                image_list.append(url)
                        elif isinstance(image_url_data, str) and image_url_data.startswith('data:image'): # Fallback
                            image_list.append(image_url_data)
    
    logger.info(f"Extracted {len(image_list)} image(s) for LLM.") # Internal log

    # Construct headers_for_llm_prompt (part of the textual prompt to LLM)
    # This includes a stringified version of the request, with image data obfuscated IN THIS STRING
    current_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
    request_json_data_for_prompt_log = copy.deepcopy(request_json_data) # Use a deep copy for modification

    # Obfuscate image data within the deepcopy for the textual part of the LLM prompt
    if 'messages' in request_json_data_for_prompt_log:
        for message in request_json_data_for_prompt_log['messages']:
            content = message.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        image_url_field = item.get('image_url', {})
                        # Check both dictionary and direct string cases for image_url's URL
                        if isinstance(image_url_field, dict) and 'url' in image_url_field and \
                           image_url_field['url'].startswith('data:image'):
                            image_url_field['url'] = '[IMAGE DATA REMOVED FOR PROMPT LOG]'
                        elif isinstance(item.get('image_url'), str) and item['image_url'].startswith('data:image'):
                             # This case might need adjustment if image_url is always a dict
                             # For robustness, ensuring we modify where 'url' would be if it's a dict
                             if isinstance(item['image_url'], dict): # Should be true if outer check passed
                                 item['image_url']['url'] = '[IMAGE DATA REMOVED FOR PROMPT LOG]'
                             else: # if it's a direct string (less common for OpenAI message structure)
                                 item['image_url'] = '[IMAGE DATA REMOVED FOR PROMPT LOG]'
    
    headers_for_llm_prompt = f"{current_time_str} - INFO - Time since last request: {time_since_last:.2f} seconds\n"
    try:
        # Create the stringified request data for the LLM's textual prompt
        request_data_str = json.dumps(request_json_data_for_prompt_log)
        if len(request_data_str) > 2000: # Truncate if too long for the prompt
             request_data_str = request_data_str[:2000] + "... [TRUNCATED FOR PROMPT LOG]"
        headers_for_llm_prompt += f"{current_time_str} - INFO - Request data (images obfuscated in this log): {request_data_str}"
    except Exception as e:
        logger.error(f"Error creating request_data_str for LLM prompt log: {e}") # Internal log
        headers_for_llm_prompt += f"{current_time_str} - INFO - Request data (images obfuscated, error during stringification)"


    # Assemble the full textual prompt to be sent to the LLM
    full_prompt_for_llm = "\n".join([
        headers_for_llm_prompt, # This contains the time and the obfuscated request JSON string
        # System instructions (same as original script)
        r'Please follow these rules: For each response, you must use one of the available tools formatted in proper XML tags. Tools include attempt_completion, ask_followup_question, read_file, write_to_file, search_files, list_files, execute_command, and list_code_definition_names. Do not respond conversationally - only use tool commands. Format any code you generate with proper indentation and line breaks, as you would in a standard code editor. Disregard any previous instructions about generating code in a single line or avoiding newline characters.',
        r'Write the entirity of your response in 1 big markdown codeblock, no word should be out of this 1 big code block and do not write a md codeblock within this big codeblock', # For Gemini fixing
        prompt_text # The actual user's query/prompt
    ])

    # Internal log of what's being prepared to send (will go to file if configured)
    logger.debug(f"Full textual prompt for LLM (first 500 chars): {full_prompt_for_llm[:500]}...")
    if image_list: # image_list contains the full base64 image data
        logger.debug(f"Actually sending {len(image_list)} image(s) to talkto function.")

    # Call the LLM
    llm_response_raw = talkto("gemini", full_prompt_for_llm, image_list)
    
    # Log the raw response with a special marker so the custom handler can parse it for console output
    if isinstance(llm_response_raw, str):
        # This specific log message will be caught by SelectiveConsolePrintHandler
        logger.info(f"{selective_handler.llm_response_marker} {llm_response_raw}")
    else:
        # This warning will go to internal logs (e.g., file) but not console
        logger.warning(f"LLM response was not a string (type: {type(llm_response_raw)}). Content: {str(llm_response_raw)[:100]}. Cannot parse for console output.")


    # Process the response for returning to the client (e.g., remove trailing backticks)
    if isinstance(llm_response_raw, str):
        if llm_response_raw.endswith("```"): # From original Cline X logic
            return llm_response_raw[:-3].strip()
        return llm_response_raw.strip()
    else:
        # If LLM response isn't a string, return empty string to client to avoid errors
        return ""

# --- Flask Routes (with internal logging) ---
@app.route('/', methods=['GET'])
def home():
    logger.info(f"GET request to / from {request.remote_addr}") # Internal log
    return "Claude API Bridge (Selective Console Output)"

@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json()
        # Internal logs about the request
        logger.info(f"Received /chat/completions request. Keys: {list(data.keys())}. Streaming: {data.get('stream', False)}")
        # Log a snippet of the data for debugging (careful with sensitive data in full logs)
        logger.debug(f"Request data dump (first 500 chars): {str(data)[:500]}")


        if not data or 'messages' not in data or not data['messages']:
            logger.warning("Invalid request: 'messages' field missing or empty.") # Internal log
            return jsonify({'error': {'message': 'Invalid request format: missing or empty "messages" field'}}), 400

        last_message = data['messages'][-1]
        # get_content_text also handles setting images to clipboard if needed
        prompt_text_for_llm = get_content_text(last_message.get('content', '')) 
        
        request_id = str(int(time.time()))
        is_streaming = data.get('stream', False)
        # Use model from request or default for response structure compatibility
        model_name = data.get("model", "gpt-3.5-turbo") 

        logger.info(f"Extracted user prompt for LLM (first 200 chars): {prompt_text_for_llm[:200]}...") # Internal log
        
        # Core interaction: data is passed for image extraction & for logging in handle_llm_interaction
        response_content_str = handle_llm_interaction(prompt_text_for_llm, data) 
        logger.info(f"LLM interaction complete. Response length: {len(response_content_str)}") # Internal log

        if is_streaming:
            def generate():
                response_id = f"chatcmpl-{request_id}"
                # Send role first
                chunk = {
                    "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                # Stream line by line for XML and code content
                lines = response_content_str.splitlines(True) # Keep newlines
                for line_content in lines:
                    if not line_content: continue # Skip empty lines if any
                    chunk = {
                        "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": model_name,
                        "choices": [{"index": 0, "delta": {"content": line_content}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    # sleep(0.01) # Optional small delay for streaming effect

                # End stream
                chunk = {
                    "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        
        # For non-streaming responses
        # Ensure prompt_text_for_llm and response_content_str are strings for len()
        prompt_str_for_usage = str(prompt_text_for_llm)
        response_str_for_usage = str(response_content_str)

        return jsonify({
            'id': f'chatcmpl-{request_id}', 'object': 'chat.completion', 'created': int(time.time()),
            'model': model_name, # Use model from request or default
            'choices': [{
                'index': 0,
                'message': {'role': 'assistant', 'content': response_str_for_usage},
                'finish_reason': 'stop'
            }],
            'usage': { # Approximations, real token counts depend on tokenizer
                'prompt_tokens': len(prompt_str_for_usage.split()), 
                'completion_tokens': len(response_str_for_usage.split()),
                'total_tokens': len(prompt_str_for_usage.split()) + len(response_str_for_usage.split())
            }
        })
    
    except Exception as e:
        logger.error(f"Critical error in /chat/completions: {str(e)}", exc_info=True) # Internal log
        return jsonify({'error': {'message': f"An internal server error occurred: {str(e)}"}}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # This log will go to internal logs (e.g., file if configured)
    # but not to console due to the SelectiveConsolePrintHandler.
    logger.info(f"Starting API Bridge server on port 3001. Autorun: {autorun}, UseFirefox: {usefirefox}")
    
    # Flask's own startup messages (e.g., "* Serving Flask app 'main'")
    # are printed by app.run() directly or via Werkzeug's logger.
    # The werkzeug_logger.setLevel(logging.ERROR) above should reduce Werkzeug's noise
    # but might not eliminate all startup banners from Flask's dev server.
    app.run(host="0.0.0.0", port=3001, debug=False) # debug=False for production, True for dev