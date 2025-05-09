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

# --- TTS Imports and Setup (New Pattern) ---
import pyttsx3
from threading import Thread, Event
import queue # For inter-thread communication
# import atexit # Could be used for a more robust shutdown hook

# --- Global TTS Variables (New Pattern) ---
tts_speech_queue = queue.Queue()
tts_worker_stop_event = Event()
tts_worker_thread_obj = None # To hold the Thread object
tts_engine_initialized_in_worker = False # Flag to check if worker has successfully initialized TTS

# --- Standard Logging Setup (configured early) ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[]
)
logger = logging.getLogger(__name__) # Main logger for the application
root_logger_obj = logging.getLogger()

for handler in root_logger_obj.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and handler.stream in (sys.stdout, sys.stderr):
        root_logger_obj.removeHandler(handler)

file_handler = logging.FileHandler("app_full.log", mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s'))
file_handler.setLevel(logging.DEBUG)
root_logger_obj.addHandler(file_handler)
# --- End Logging Setup ---


# --- TTS Worker Thread Function (New Pattern) ---
def tts_worker_function():
    global tts_engine_initialized_in_worker # To update the global flag
    
    engine = None # Engine is local to this thread
    init_attempts_thread = 0
    logger.info("TTS_WORKER: Thread started.")

    try:
        init_attempts_thread += 1
        logger.info(f"TTS_WORKER: Initializing TTS engine (Attempt: {init_attempts_thread})...")
        if sys.platform == "win32":
            engine = pyttsx3.init(driverName='sapi5')
        else:
            engine = pyttsx3.init()

        if engine:
            tts_engine_initialized_in_worker = True # Signal successful initialization
            logger.info(f"TTS_WORKER: pyttsx3.init() successful. Engine object ID: {id(engine)}")
            
            # Optional: Speak a confirmation from the worker itself
            engine.say("TTS service is now active.")
            engine.runAndWait()
            logger.info("TTS_WORKER: Spoke its initialization message.")
        else:
            logger.error("TTS_WORKER: pyttsx3.init() returned None. Worker cannot function.")
            tts_engine_initialized_in_worker = False
            return # Exit thread if engine fails

        # Main loop: process speech requests from the queue
        while not tts_worker_stop_event.is_set():
            try:
                text_to_say = tts_speech_queue.get(timeout=0.2) # Short timeout to check stop_event
                if text_to_say is None: # Sentinel to stop
                    logger.info("TTS_WORKER: Received None (sentinel) from queue. Exiting loop.")
                    tts_worker_stop_event.set() # Ensure loop terminates if not already set
                    break
                
                logger.info(f"TTS_WORKER: Got from queue: '{text_to_say[:100]}'. Speaking...")
                engine.say(str(text_to_say))
                engine.runAndWait()
                logger.info(f"TTS_WORKER: Finished speaking: '{text_to_say[:100]}'.")
                tts_speech_queue.task_done()
            except queue.Empty:
                continue # Timeout, just check stop_event again
            except Exception as e:
                logger.error(f"TTS_WORKER: Error during speech processing: {e}", exc_info=True)
                # Depending on error, might want to re-init or break
                # For now, log and continue
        
    except Exception as e_outer:
        logger.error(f"TTS_WORKER: Major error in worker thread: {e_outer}", exc_info=True)
        tts_engine_initialized_in_worker = False # Mark as failed
    finally:
        if engine and hasattr(engine, 'stop'):
            logger.info("TTS_WORKER: Attempting to stop engine in finally block (if supported).")
            # engine.stop() # Often not needed for SAPI5 with runAndWait(), can be problematic.
        logger.info("TTS_WORKER: Thread finished.")

# --- TTS Control Functions (New Pattern) ---
def start_tts_service():
    global tts_worker_thread_obj, tts_engine_initialized_in_worker
    if tts_worker_thread_obj is not None and tts_worker_thread_obj.is_alive():
        logger.info("TTS service: Worker thread already running.")
        return

    logger.info("TTS service: Starting worker thread...")
    tts_worker_stop_event.clear() # Reset for a potential restart
    tts_engine_initialized_in_worker = False # Will be set by worker

    tts_worker_thread_obj = Thread(target=tts_worker_function, daemon=True)
    tts_worker_thread_obj.start()
    logger.info("TTS service: Worker thread has been initiated.")
    # It's good to wait a very short moment to let the thread actually start
    # and potentially initialize before the main app tries to use it too quickly.
    time.sleep(0.5) # Give thread a moment to spin up

def stop_tts_service():
    global tts_worker_thread_obj
    logger.info("TTS service: Requesting worker thread to stop...")
    tts_worker_stop_event.set() # Signal the loop to exit
    if tts_worker_thread_obj is not None and tts_worker_thread_obj.is_alive():
        logger.debug("TTS service: Putting None sentinel on queue for graceful shutdown.")
        tts_speech_queue.put(None) # Ensure .get() doesn't block indefinitely
        tts_worker_thread_obj.join(timeout=5) # Wait for thread to finish
        if tts_worker_thread_obj.is_alive():
            logger.warning("TTS service: Worker thread did not stop within timeout.")
        else:
            logger.info("TTS service: Worker thread stopped successfully.")
    else:
        logger.info("TTS service: Worker thread was not running or already stopped.")
    tts_worker_thread_obj = None # Clear the thread object

def speak_via_queue(text_to_say: str):
    global tts_engine_initialized_in_worker
    
    if tts_worker_thread_obj is None or not tts_worker_thread_obj.is_alive():
        logger.error("TTS speak_via_queue: Worker thread is not running! Attempting to restart.")
        start_tts_service()
        time.sleep(2) # Give it more time to initialize if restarted
        if not tts_engine_initialized_in_worker:
            logger.error("TTS speak_via_queue: Worker failed to restart or initialize. Speech will be lost for this request.")
            return

    if not tts_engine_initialized_in_worker: # Check after potential restart
        logger.warning(f"TTS speak_via_queue: Worker not confirmed initialized. Queuing '{text_to_say[:30]}...', but speech depends on successful worker init.")
        # The message is queued; if worker init succeeds later, it will be spoken.
        # If init failed, worker thread would have exited, and message remains on queue (or lost if app restarts).

    text_to_say_cleaned = str(text_to_say).strip()
    if not text_to_say_cleaned:
        logger.debug("speak_via_queue: Cleaned text is empty, not queueing.")
        return
    
    logger.debug(f"speak_via_queue: Putting on queue: '{text_to_say_cleaned[:100]}'")
    tts_speech_queue.put(text_to_say_cleaned)


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

        thinking_content, result_content, ask_followup_raw_content = None, None, None
        try:
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
            if thinking_match: thinking_content = thinking_match.group(1).strip()
            attempt_completion_match = re.search(r"<attempt_completion>(.*?)</attempt_completion>", response_text, re.DOTALL)
            if attempt_completion_match:
                result_tag_match = re.search(r"<result>(.*?)</result>", attempt_completion_match.group(1), re.DOTALL)
                if result_tag_match: result_content = result_tag_match.group(1).strip()
            ask_followup_match = re.search(r"<ask_followup_question>(.*?)</ask_followup_question>", response_text, re.DOTALL)
            if ask_followup_match: ask_followup_raw_content = ask_followup_match.group(1).strip()
        except Exception as e:
            logger.error(f"SelectiveConsolePrintHandler: Regex parsing error: {e}", exc_info=True)

        console_output_blocks, speech_parts_to_say = [], []
        if thinking_content:
            console_output_blocks.append(thinking_content)
            speech_parts_to_say.append(thinking_content)
        if result_content:
            console_output_blocks.append(f"Attempt Completion, {result_content}")
            speech_parts_to_say.append(f"Attempt Completion. {result_content}")
        if ask_followup_raw_content:
            question_text_parsed, options_list_str_parsed = "", ""
            question_text_match = re.search(r"<question>(.*?)</question>", ask_followup_raw_content, re.DOTALL)
            if question_text_match: question_text_parsed = question_text_match.group(1).strip()
            options_text_match = re.search(r"<options>(.*?)</options>", ask_followup_raw_content, re.DOTALL)
            if options_text_match:
                raw_options_str = options_text_match.group(1).strip()
                try:
                    options_parsed_json = json.loads(raw_options_str)
                    if isinstance(options_parsed_json, list):
                        options_list_str_parsed = ", ".join(str(opt) for opt in options_parsed_json)
                except: # Simplified fallback
                    cleaned_options_val = raw_options_str.strip("[]")
                    parts = [opt.strip().strip('"\'') for opt in cleaned_options_val.split(',')]
                    options_list_str_parsed = ", ".join(p for p in parts if p)
            
            followup_console_parts = []
            if question_text_parsed:
                followup_console_parts.append(question_text_parsed)
                speech_parts_to_say.append(f"I have a question: {question_text_parsed}")
            if options_list_str_parsed:
                followup_console_parts.append(options_list_str_parsed)
                speech_parts_to_say.append(f"Your options are: {options_list_str_parsed}")
            if followup_console_parts:
                console_output_blocks.append("\n".join(followup_console_parts))

        if console_output_blocks:
            sys.stdout.write("\n\n".join(filter(None, console_output_blocks)) + "\n")
            sys.stdout.flush()
        if speech_parts_to_say:
            full_speech_output = ". ".join(filter(None, speech_parts_to_say)).strip()
            if full_speech_output:
                speak_via_queue(full_speech_output) # MODIFIED HERE
            else:
                logger.debug("SelectiveConsolePrintHandler: full_speech_output empty.")
        else:
            logger.debug("SelectiveConsolePrintHandler: speech_parts_to_say empty.")

selective_handler = SelectiveConsolePrintHandler()
selective_handler.setFormatter(logging.Formatter('%(message)s')) 
selective_handler.setLevel(logging.INFO) 
root_logger_obj.addHandler(selective_handler)

try:
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.ERROR); werkzeug_logger.propagate = False 
except Exception as e: logger.warning(f"Werkzeug logger config error: {e}")

app = Flask(__name__)
last_request_time = 0
MIN_REQUEST_INTERVAL = 5 
set_autopath(r"D:\cline-x-claudeweb\images")
set_altpath(r"D:\cline-x-claudeweb\images\alt1440")

# --- Clipboard, Content Extraction, LLM Interaction, Flask Routes (Largely Unchanged) ---
# (Ensure they use logger correctly if needed, but their core logic for TTS is now via speak_via_queue)

def set_clipboard(text, retries=3, delay=0.2):
    for i in range(retries):
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            try: win32clipboard.SetClipboardText(str(text))
            except: win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, str(text).encode('utf-16le'))
            win32clipboard.CloseClipboard()
            logger.debug("Set clipboard text successfully.")
            return
        except pywintypes.error as e:
            if e.winerror == 5: logger.warning(f"Clipboard access denied. Retrying... ({i+1}/{retries})"); time.sleep(delay)
            else: logger.error(f"pywintypes.error setting clipboard: {e}", exc_info=True); raise
        except Exception as e: logger.error(f"Exception setting clipboard: {e}", exc_info=True); raise
    logger.error(f"Failed to set clipboard text after {retries} attempts.")

def set_clipboard_image(image_data):
    try:
        binary_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(binary_data))
        output = io.BytesIO(); image.convert("RGB").save(output, "BMP"); data = output.getvalue()[14:]; output.close()
        win32clipboard.OpenClipboard(); win32clipboard.EmptyClipboard(); win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data); win32clipboard.CloseClipboard()
        logger.debug("Set image to clipboard.")
        return True
    except Exception as e: logger.error(f"Error setting image to clipboard: {e}", exc_info=True); return False

def get_content_text(content: Union[str, List[Dict[str, str]], Dict[str, str]]) -> str:
    if isinstance(content, str): return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if item.get("type") == "text": parts.append(item["text"])
            elif item.get("type") == "image_url":
                image_data = item.get("image_url", {}).get("url", "")
                if image_data.startswith('data:image'): set_clipboard_image(image_data)
                parts.append(f"[Image: {item.get('description', 'Uploaded image')}]")
            elif item.get("type") == "image":
                image_data_url = item.get("image_url", {}).get("url", "")
                if not image_data_url and "data" in item:
                    try: image_data_url = 'data:image/unknown;base64,' + base64.b64encode(item["data"] if isinstance(item["data"], bytes) else item["data"].encode('utf-8')).decode('utf-8')
                    except: image_data_url = ""
                if image_data_url.startswith('data:image'): set_clipboard_image(image_data_url)
                parts.append(f"[Image: {item.get('description', 'Uploaded image')}]")
        return "\n".join(parts)
    elif isinstance(content, dict):
        if content.get("type") == "image":
            image_data_url = content.get("image_url", {}).get("url", "")
            if not image_data_url and "data" in content:
                try: image_data_url = 'data:image/unknown;base64,' + base64.b64encode(content["data"] if isinstance(content["data"], bytes) else content["data"].encode('utf-8')).decode('utf-8')
                except: image_data_url = ""
            if image_data_url.startswith('data:image'): set_clipboard_image(image_data_url)
            return f"[Image: {content.get('description', 'Uploaded image')}]"
        return content.get("text", "")
    return ""

def handle_llm_interaction(prompt_text: str, request_json_data: Dict):
    global last_request_time
    logger.info(f"LLM: User prompt (first 200): {prompt_text[:200]}...")
    current_time = time.time()
    if current_time - last_request_time < MIN_REQUEST_INTERVAL:
        sleep(MIN_REQUEST_INTERVAL - (current_time - last_request_time))
    last_request_time = time.time()
    image_list = []
    if 'messages' in request_json_data:
        for message in request_json_data['messages']:
            content = message.get('content', [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        url_data = item.get('image_url', {})
                        url = url_data.get('url', '') if isinstance(url_data, dict) else (url_data if isinstance(url_data, str) else "")
                        if url.startswith('data:image'): image_list.append(url)
    
    log_data = copy.deepcopy(request_json_data)
    if 'messages' in log_data:
        for msg in log_data['messages']:
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        if isinstance(item.get('image_url'), dict) and 'url' in item['image_url']: item['image_url']['url'] = '[IMG_REMOVED]'
                        elif isinstance(item.get('image_url'), str): item['image_url'] = '[IMG_REMOVED_STR]'
    
    header = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Request data: {json.dumps(log_data)}"
    prompt_parts = [header, 
                    r'Please follow these rules: For each response, you must use one of conseguavailable tools formatted in proper XML tags. Tools include attempt_completion, ask_followup_question, read_file, write_to_file, search_files, list_files, execute_command, and list_code_definition_names. Do not respond conversationally - only use tool commands. Format any code you generate with proper indentation and line breaks, as you would in a standard code editor. Disregard any previous instructions about generating code in a single line or avoiding newline characters.',
                    r'Write the entirity of your response in 1 big markdown codeblock, no word should be out of this 1 big code block and do not write a md codeblock within this big codeblock',
                    prompt_text]
    full_prompt = "\n".join(prompt_parts)
    
    llm_response_raw = talkto("gemini", full_prompt, image_list)
    
    if isinstance(llm_response_raw, str):
        logger.info(f"{selective_handler.llm_response_marker} {llm_response_raw}")
    else:
        logger.warning(f"LLM response not str: {type(llm_response_raw)}. Content: {str(llm_response_raw)[:100]}.")
        return ""

    if isinstance(llm_response_raw, str):
        processed = llm_response_raw.strip()
        if processed.startswith("```") and processed.endswith("```"):
            processed = re.sub(r'^```[a-zA-Z]*\n?', '', processed)
            processed = re.sub(r'\n?```$', '', processed).strip()
        elif processed.endswith("```"): processed = processed[:-3].strip()
        return processed
    return ""

@app.route('/', methods=['GET'])
def home():
    logger.info(f"GET / from {request.remote_addr}")
    return "API Bridge: TTS with Single Worker Queue"

@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json()
        logger.info(f"POST /chat/completions. Streaming: {data.get('stream', False)}")
        if not data or 'messages' not in data or not data['messages']:
            return jsonify({'error': {'message': 'Invalid request: "messages" missing/empty'}}), 400
        
        prompt_text = get_content_text(data['messages'][-1].get('content', ''))
        request_id, model_name = str(int(time.time())), data.get("model", "gpt-3.5-turbo")
        
        response_str = handle_llm_interaction(prompt_text, data)
        logger.info(f"LLM response len: {len(response_str)}")

        if data.get('stream', False):
            def generate_stream():
                yield f"data: {json.dumps({'id': f'chatcmpl-{request_id}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
                for line in response_str.splitlines(True):
                    if not line: continue
                    yield f"data: {json.dumps({'id': f'chatcmpl-{request_id}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {'content': line}, 'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({'id': f'chatcmpl-{request_id}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(generate_stream(), mimetype='text/event-stream')
        
        return jsonify({'id': f'chatcmpl-{request_id}', 'object': 'chat.completion', 'created': int(time.time()), 'model': model_name, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': response_str}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': len(prompt_text.split()), 'completion_tokens': len(response_str.split()), 'total_tokens': len(prompt_text.split()) + len(response_str.split())}})
    except Exception as e:
        logger.error(f"Error in /chat/completions: {e}", exc_info=True)
        return jsonify({'error': {'message': f"Internal server error: {e}"}}), 500

# --- Main Execution ---
if __name__ == '__main__':
    start_tts_service() # Start the TTS worker thread

    # Wait a bit to ensure TTS worker initializes and speaks its intro.
    # This also helps confirm tts_engine_initialized_in_worker is set.
    logger.info("Main: Waiting for TTS worker to potentially speak its init message...")
    time.sleep(3) # Adjust as needed, depends on how fast your TTS init is.

    startup_message = "Cline x voice running"
    sys.stdout.write(startup_message + "\n")
    sys.stdout.flush() 
    
    if tts_engine_initialized_in_worker:
        logger.info("Main: Attempting to speak startup message via queue...")
        speak_via_queue(startup_message) 
    else:
        logger.warning("Main: TTS worker not confirmed initialized, skipping startup message speech via queue.")
    
    logger.info(f"Main: Starting API Bridge server on port 3001. TTS Worker Initialized Flag: {tts_engine_initialized_in_worker}")
    
    try:
        app.run(host="0.0.0.0", port=3001, debug=False, use_reloader=False)
    finally:
        logger.info("Main: Application shutting down...")
        stop_tts_service() # Request TTS worker to stop and join
        logger.info("Main: Application shutdown complete.")