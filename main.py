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
        
        printed_anything_to_console = False

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

        if thinking_content:
            sys.stdout.write("Thinking,\n") # Changed
            sys.stdout.write(thinking_content + "\n")
            printed_anything_to_console = True

        if result_content:
            if printed_anything_to_console:
                sys.stdout.write("\n")
            sys.stdout.write(f"Attempt Completion, {result_content}\n") # Changed
            printed_anything_to_console = True
        
        if ask_followup_raw_content:
            if printed_anything_to_console:
                sys.stdout.write("\n")

            question_text = ""
            options_list_str = ""
            
            question_text_match = re.search(r"<question>(.*?)</question>", ask_followup_raw_content, re.DOTALL)
            if question_text_match:
                question_text = question_text_match.group(1).strip()
            
            options_text_match = re.search(r"<options>(.*?)</options>", ask_followup_raw_content, re.DOTALL)
            if options_text_match:
                raw_options_str = options_text_match.group(1).strip()
                try:
                    # Attempt to parse as JSON list first
                    options_parsed = json.loads(raw_options_str)
                    if isinstance(options_parsed, list):
                        options_list_str = ", ".join(str(opt) for opt in options_parsed)
                except json.JSONDecodeError:
                    # Fallback: basic cleaning if not valid JSON
                    cleaned_options_val = raw_options_str
                    # Remove potential surrounding brackets
                    if cleaned_options_val.startswith('[') and cleaned_options_val.endswith(']'):
                        cleaned_options_val = cleaned_options_val[1:-1]
                    
                    # Split by comma, then strip whitespace and quotes from each part
                    parts = [opt.strip().strip('"').strip("'") for opt in cleaned_options_val.split(',')]
                    options_list_str = ", ".join(p for p in parts if p) # Filter out empty strings
                except Exception:
                    # Catch any other error during options processing
                    options_list_str = "" # Default to empty if complex error

            followup_block_printed_content = False
            if question_text:
                sys.stdout.write(f"I have question, {question_text}\n")
                followup_block_printed_content = True
            
            if options_list_str:
                # Options follow directly after the question or start if no question
                sys.stdout.write(options_list_str + "\n")
                followup_block_printed_content = True
            
            if followup_block_printed_content:
                printed_anything_to_console = True # Mark that this block printed something

        if printed_anything_to_console:
            sys.stdout.flush()

# --- Standard Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[] # Start with no handlers, we will add ours
)
logger = logging.getLogger(__name__)

root_logger = logging.getLogger()
# Remove any default console handlers that might have been added by basicConfig or Flask
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
        root_logger.removeHandler(handler)

# Add our custom selective console handler
selective_handler = SelectiveConsolePrintHandler()
selective_handler.setFormatter(logging.Formatter('%(message)s')) # Only message for this handler
selective_handler.setLevel(logging.INFO) # Process INFO and above for LLM responses
root_logger.addHandler(selective_handler)

# Optionally, add a FileHandler for full logs
# file_handler = logging.FileHandler("app_full.log", mode='w') 
# file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'))
# file_handler.setLevel(logging.DEBUG)
# root_logger.addHandler(file_handler)

# Suppress Werkzeug's default logging to keep console clean for our output
try:
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.ERROR) # Show only errors from Werkzeug
    # Prevent werkzeug logs from propagating to the root logger's handlers
    werkzeug_logger.propagate = False 
except Exception as e:
    logger.warning(f"Could not modify werkzeug logger settings: {e}")


# --- Flask App and Globals ---
app = Flask(__name__)
last_request_time = 0
MIN_REQUEST_INTERVAL = 5 # seconds

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
            except Exception: # Fallback for potential encoding issues with SetClipboardText
                win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, str(text).encode('utf-16le'))
            win32clipboard.CloseClipboard()
            logger.debug("Set clipboard text successfully.")
            return
        except pywintypes.error as e:
            if e.winerror == 5: # Access is denied
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
        # Assuming image_data is a base64 string like "data:image/png;base64,iVBORw0KGgo..."
        binary_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(binary_data))
        
        # Convert to BMP format for clipboard
        output = io.BytesIO()
        image.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:] # Skip BMP file header
        output.close()
        
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data) # CF_DIB for device-independent bitmap
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
            elif item.get("type") == "image_url": # OpenAI format
                image_data = item.get("image_url", {}).get("url", "")
                if image_data.startswith('data:image'):
                    logger.info(f"Found image data in content list (item {item_idx}, type 'image_url'), attempting to set to clipboard.")
                    set_clipboard_image(image_data)
                description = item.get("description", "An uploaded image") # Custom field for description
                parts.append(f"[Image: {description}]")
            elif item.get("type") == "image": # Anthropic format (or similar)
                # Handle different ways image data might be structured
                image_data_url = item.get("image_url", {}).get("url", "") # If nested like OpenAI
                if not image_data_url and "data" in item: # If data is directly in item
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
    elif isinstance(content, dict): # Single content item that is a dictionary
        text_content = content.get("text", "")
        if content.get("type") == "image": # If the dict itself is an image item
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
            return f"[Image: {description}]" # Return only image placeholder
        return text_content # If not an image type, return text content
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

    # Extract images for the LLM
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
                        elif isinstance(image_url_data, str) and image_url_data.startswith('data:image'): # If 'image_url' is just the data string
                             image_list.append(image_url_data)
    
    logger.info(f"Extracted {len(image_list)} image(s) for LLM.")

    # Prepare request data for logging (obfuscate images)
    current_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
    request_json_data_for_prompt_log = copy.deepcopy(request_json_data)

    if 'messages' in request_json_data_for_prompt_log:
        for message in request_json_data_for_prompt_log['messages']:
            content = message.get('content', [])
            if isinstance(content, list):
                for item_idx, item in enumerate(content): # Use enumerate for potential detailed logging
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        image_url_field = item.get('image_url', {})
                        obfuscation_str = '[IMAGE DATA REMOVED]'
                        if isinstance(image_url_field, dict) and 'url' in image_url_field and \
                           image_url_field['url'].startswith('data:image'):
                            image_url_field['url'] = obfuscation_str
                        elif isinstance(item.get('image_url'), str) and item['image_url'].startswith('data:image'): # Handle if image_url is a string
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
    # Note: 'autorun' related prompt part is removed as autorun functionality is removed.
    
    prompt_parts.append(r'Please follow these rules: For each response, you must use one of conseguavailable tools formatted in proper XML tags. Tools include attempt_completion, ask_followup_question, read_file, write_to_file, search_files, list_files, execute_command, and list_code_definition_names. Do not respond conversationally - only use tool commands. Format any code you generate with proper indentation and line breaks, as you would in a standard code editor. Disregard any previous instructions about generating code in a single line or avoiding newline characters.')
    prompt_parts.append(r'Write the entirity of your response in 1 big markdown codeblock, no word should be out of this 1 big code block and do not write a md codeblock within this big codeblock') # For Gemini fixing
    prompt_parts.append(prompt_text)

    full_prompt_for_llm = "\n".join(prompt_parts)

    logger.debug(f"Full textual prompt for LLM (first 500 chars): {full_prompt_for_llm[:500]}...")
    if image_list:
        logger.debug(f"Actually sending {len(image_list)} image(s) to talkto function.")

    llm_response_raw = talkto("gemini", full_prompt_for_llm, image_list)
    
    # Log the raw response for parsing by our custom handler
    if isinstance(llm_response_raw, str):
        logger.info(f"{selective_handler.llm_response_marker} {llm_response_raw}")
    else:
        logger.warning(f"LLM response was not a string (type: {type(llm_response_raw)}). Content: {str(llm_response_raw)[:100]}. Cannot parse for console output.")

    # Process response: strip markdown backticks if present
    if isinstance(llm_response_raw, str):
        processed_response = llm_response_raw.strip()
        if processed_response.startswith("```") and processed_response.endswith("```"):
             # More robust stripping of triple backticks and potential language specifier
            processed_response = re.sub(r'^```[a-zA-Z]*\n?', '', processed_response)
            processed_response = re.sub(r'\n?```$', '', processed_response)
            processed_response = processed_response.strip()
        elif processed_response.endswith("```"): # If only trailing backticks
            processed_response = processed_response[:-3].strip()
        return processed_response
    else:
        return "" # Return empty string if response is not string

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

        # Extract the last user message content to form the main prompt text
        last_message = data['messages'][-1]
        # get_content_text will also handle image data within the message for clipboard
        prompt_text_for_llm = get_content_text(last_message.get('content', '')) 
        
        request_id = str(int(time.time())) # Simple request ID
        is_streaming = data.get('stream', False)
        model_name = data.get("model", "gpt-3.5-turbo") # Use requested model or a default

        logger.info(f"Extracted user prompt for LLM (first 200 chars): {prompt_text_for_llm[:200]}...")
        
        # Get response from LLM
        response_content_str = handle_llm_interaction(prompt_text_for_llm, data) 
        logger.info(f"LLM interaction complete. Response length: {len(response_content_str)}")

        if is_streaming:
            def generate():
                response_id = f"chatcmpl-{request_id}"
                # Send initial chunk with role
                chunk = {
                    "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                # Stream content line by line
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

                # Send final chunk with finish reason
                chunk = {
                    "id": response_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return Response(generate(), mimetype='text/event-stream')
        
        # Non-streaming response
        # Ensure string types for usage calculation
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
            'usage': { # Simple token count based on space-separated words
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
    # Print custom startup message directly to stdout
    sys.stdout.write("Cline x voice running\n")
    sys.stdout.flush() 
    
    # This log message will go to file if file handler is enabled, not to console due to selective handler
    logger.info(f"Starting API Bridge server on port 3001.")
    
    app.run(host="0.0.0.0", port=3001, debug=False)