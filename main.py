from flask import Flask, jsonify, request, Response
import win32clipboard # Assuming this is for Windows clipboard operations
import time
import pywintypes # For win32clipboard errors
from time import sleep
# from optimisewait import set_autopath, set_altpath # Assuming this exists
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
from threading import Thread, Event, Lock 
import queue 

# --- Global TTS Variables (New Pattern) ---
tts_speech_queue = queue.Queue()
tts_worker_stop_event = Event()
tts_worker_thread_obj = None
tts_engine_initialized_in_worker = False

# --- Global State for Readiness ---
llm_interaction_lock = Lock()
is_processing_llm_request = False
# Status for /status endpoint: "IDLE", "PROCESSING", "AWAITING_FOLLOWUP", "COMPLETED_NO_FOLLOWUP", "ERROR"
current_system_status = "IDLE" 
system_status_message = "System initializing..." # For /status endpoint

# --- Standard Logging Setup (configured early) ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[]
)
logger = logging.getLogger(__name__)
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
    global tts_engine_initialized_in_worker
    engine = None
    init_attempts_thread = 0
    logger.info("TTS_WORKER: Thread started.")
    try:
        init_attempts_thread += 1
        logger.info(f"TTS_WORKER: Initializing TTS engine (Attempt: {init_attempts_thread})...")
        if sys.platform == "win32": engine = pyttsx3.init(driverName='sapi5')
        else: engine = pyttsx3.init()

        if engine:
            tts_engine_initialized_in_worker = True
            logger.info(f"TTS_WORKER: pyttsx3.init() successful. Engine object ID: {id(engine)}")
            engine.say("TTS service is now active.")
            engine.runAndWait()
            logger.info("TTS_WORKER: Spoke its initialization message.")
        else:
            logger.error("TTS_WORKER: pyttsx3.init() returned None. Worker cannot function.")
            tts_engine_initialized_in_worker = False
            return

        while not tts_worker_stop_event.is_set():
            try:
                text_to_say = tts_speech_queue.get(timeout=0.2)
                if text_to_say is None:
                    logger.info("TTS_WORKER: Received None (sentinel) from queue. Exiting loop.")
                    tts_worker_stop_event.set()
                    break
                logger.info(f"TTS_WORKER: Got from queue: '{text_to_say[:100]}'. Speaking...")
                engine.say(str(text_to_say))
                engine.runAndWait()
                logger.info(f"TTS_WORKER: Finished speaking: '{text_to_say[:100]}'.")
                tts_speech_queue.task_done()
            except queue.Empty: continue
            except Exception as e: logger.error(f"TTS_WORKER: Error during speech processing: {e}", exc_info=True)
    except Exception as e_outer:
        logger.error(f"TTS_WORKER: Major error in worker thread: {e_outer}", exc_info=True)
        tts_engine_initialized_in_worker = False
    finally:
        if engine and hasattr(engine, 'stop'): logger.info("TTS_WORKER: Attempting to stop engine (if supported).")
        logger.info("TTS_WORKER: Thread finished.")

# --- TTS Control Functions (New Pattern) ---
def start_tts_service():
    global tts_worker_thread_obj, tts_engine_initialized_in_worker
    if tts_worker_thread_obj is not None and tts_worker_thread_obj.is_alive():
        logger.info("TTS service: Worker thread already running.")
        return
    logger.info("TTS service: Starting worker thread...")
    tts_worker_stop_event.clear()
    tts_engine_initialized_in_worker = False
    tts_worker_thread_obj = Thread(target=tts_worker_function, daemon=True)
    tts_worker_thread_obj.start()
    logger.info("TTS service: Worker thread has been initiated.")
    time.sleep(0.5)

def stop_tts_service():
    global tts_worker_thread_obj
    logger.info("TTS service: Requesting worker thread to stop...")
    tts_worker_stop_event.set()
    if tts_worker_thread_obj is not None and tts_worker_thread_obj.is_alive():
        logger.debug("TTS service: Putting None sentinel on queue for graceful shutdown.")
        tts_speech_queue.put(None)
        tts_worker_thread_obj.join(timeout=5)
        if tts_worker_thread_obj.is_alive(): logger.warning("TTS service: Worker thread did not stop within timeout.")
        else: logger.info("TTS service: Worker thread stopped successfully.")
    else: logger.info("TTS service: Worker thread was not running or already stopped.")
    tts_worker_thread_obj = None

def speak_via_queue(text_to_say: str):
    global tts_engine_initialized_in_worker
    if tts_worker_thread_obj is None or not tts_worker_thread_obj.is_alive():
        logger.error("TTS speak_via_queue: Worker thread is not running! Attempting to restart.")
        start_tts_service(); time.sleep(2)
        if not tts_engine_initialized_in_worker:
            logger.error("TTS speak_via_queue: Worker failed to restart. Speech lost.")
            return
    if not tts_engine_initialized_in_worker:
        logger.warning(f"TTS speak_via_queue: Worker not confirmed initialized. Queuing '{text_to_say[:30]}...'")
    text_to_say_cleaned = str(text_to_say).strip()
    if not text_to_say_cleaned:
        logger.debug("speak_via_queue: Cleaned text is empty.")
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
        if self.llm_response_marker not in log_message: return
        logger.debug(f"SelectiveConsolePrintHandler received LLM log: {log_message[:200]}...")
        try: response_text = log_message.split(self.llm_response_marker, 1)[1].strip()
        except IndexError:
            logger.warning("SelectiveConsolePrintHandler: Could not split marker."); return

        thinking_content, result_content, ask_followup_raw_content = None, None, None
        try:
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", response_text, re.DOTALL)
            if thinking_match: thinking_content = thinking_match.group(1).strip()
            attempt_match = re.search(r"<attempt_completion>(.*?)</attempt_completion>", response_text, re.DOTALL)
            if attempt_match:
                result_match = re.search(r"<result>(.*?)</result>", attempt_match.group(1), re.DOTALL)
                if result_match: result_content = result_match.group(1).strip()
            ask_match = re.search(r"<ask_followup_question>(.*?)</ask_followup_question>", response_text, re.DOTALL)
            if ask_match: ask_followup_raw_content = ask_match.group(1).strip()
        except Exception as e: logger.error(f"SelectiveConsolePrintHandler: Regex error: {e}", exc_info=True); return

        console_blocks, speech_parts = [], []
        if thinking_content: console_blocks.append(thinking_content)
        if result_content:
            console_blocks.append(f"Attempt Completion, {result_content}")
            speech_parts.append(f"Attempt Completion. {result_content}")
        if ask_followup_raw_content:
            q_text, opts_str = "", ""
            q_match = re.search(r"<question>(.*?)</question>", ask_followup_raw_content, re.DOTALL)
            if q_match: q_text = q_match.group(1).strip()
            opts_match = re.search(r"<options>(.*?)</options>", ask_followup_raw_content, re.DOTALL)
            if opts_match:
                raw_opts = opts_match.group(1).strip()
                try:
                    opts_json = json.loads(raw_opts)
                    if isinstance(opts_json, list): opts_str = ", ".join(map(str, opts_json))
                except: opts_str = ", ".join(p.strip().strip('"\'') for p in raw_opts.strip("[]").split(',') if p.strip())
            
            followup_console = []
            if q_text: 
                followup_console.append(q_text); speech_parts.append(f"I have a question: {q_text}")
            if opts_str:
                followup_console.append(opts_str); speech_parts.append(f"Your options are: {opts_str}")
            if followup_console: console_blocks.append("\n".join(followup_console))

        if console_blocks:
            sys.stdout.write("\n\n" + "\n\n".join(filter(None, console_blocks)) + "\n")
            sys.stdout.flush()
        if speech_parts:
            speech_out = ". ".join(filter(None, speech_parts)).strip()
            if speech_out: speak_via_queue(speech_out)
            # else: logger.debug("SelectiveConsolePrintHandler: speech_out empty.")
        # else: logger.debug("SelectiveConsolePrintHandler: speech_parts empty.")

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
MIN_REQUEST_INTERVAL = 1

try:
    from optimisewait import set_autopath, set_altpath
    set_autopath(r"D:\cline-x-claudeweb\images") 
    set_altpath(r"D:\cline-x-claudeweb\images\alt1440")
except (ImportError, NameError) as e:
    logger.warning(f"optimisewait functions not available: {e}. Skipping.")

def set_clipboard(text, retries=3, delay=0.2):
    for i in range(retries):
        try:
            win32clipboard.OpenClipboard(); win32clipboard.EmptyClipboard()
            try: win32clipboard.SetClipboardText(str(text))
            except: win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, str(text).encode('utf-16le'))
            win32clipboard.CloseClipboard(); logger.debug("Set clipboard text."); return
        except pywintypes.error as e:
            if e.winerror == 5: logger.warning(f"Clipboard access denied. Retry {i+1}"); time.sleep(delay)
            else: logger.error(f"pywintypes error: {e}", exc_info=True); raise
        except Exception as e: logger.error(f"Clipboard error: {e}", exc_info=True); raise
    logger.error(f"Failed to set clipboard after {retries} retries.")

def set_clipboard_image(image_data):
    try:
        img_bytes = base64.b64decode(image_data.split(',')[1])
        img = Image.open(io.BytesIO(img_bytes)); output = io.BytesIO()
        img.convert("RGB").save(output, "BMP"); data = output.getvalue()[14:]; output.close()
        win32clipboard.OpenClipboard(); win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data); win32clipboard.CloseClipboard()
        logger.debug("Set image to clipboard."); return True
    except Exception as e: logger.error(f"Set clipboard image error: {e}", exc_info=True); return False

def get_content_text(content: Union[str, List[Dict[str, str]], Dict[str, str]]) -> str:
    if isinstance(content, str): return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if item.get("type") == "text": parts.append(item["text"])
            elif item.get("type") == "image_url":
                img_url = item.get("image_url", {}).get("url", "")
                if img_url.startswith('data:image'): set_clipboard_image(img_url)
                parts.append(f"[Image: {item.get('description', 'Uploaded image')}]")
            elif item.get("type") == "image":
                img_url = item.get("image_url", {}).get("url", "")
                if not img_url and "data" in item:
                    try: 
                        b64data = item["data"] if isinstance(item["data"],bytes) else item["data"].encode()
                        img_url = 'data:image/unknown;base64,' + base64.b64encode(b64data).decode()
                    except: img_url = ""
                if img_url.startswith('data:image'): set_clipboard_image(img_url)
                parts.append(f"[Image: {item.get('description', 'Uploaded image')}]")
        return "\n".join(parts)
    elif isinstance(content, dict):
        if content.get("type") == "image":
            img_url = content.get("image_url", {}).get("url", "")
            if not img_url and "data" in content:
                try: 
                    b64data = content["data"] if isinstance(content["data"],bytes) else content["data"].encode()
                    img_url = 'data:image/unknown;base64,' + base64.b64encode(b64data).decode()
                except: img_url = ""
            if img_url.startswith('data:image'): set_clipboard_image(img_url)
            return f"[Image: {content.get('description', 'Uploaded image')}]"
        return content.get("text", "")
    return ""

def handle_llm_interaction(prompt_text: str, request_json_data: Dict) -> Dict:
    global last_request_time
    logger.info(f"LLM: User prompt (200): {prompt_text[:200]}...")
    current_time = time.time()
    if current_time - last_request_time < MIN_REQUEST_INTERVAL:
        sleep(MIN_REQUEST_INTERVAL - (current_time - last_request_time))
    last_request_time = time.time()
    
    images = []
    if 'messages' in request_json_data:
        for msg_content in request_json_data['messages']:
            content_items = msg_content.get('content', [])
            if isinstance(content_items, list):
                for item in content_items:
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        url_data = item.get('image_url', {})
                        url = url_data.get('url', '') if isinstance(url_data,dict) else (url_data if isinstance(url_data,str) else "")
                        if url.startswith('data:image'): images.append(url)
    
    log_data = copy.deepcopy(request_json_data)
    if 'messages' in log_data:
        for msg in log_data['messages']:
            if isinstance(msg.get('content'), list):
                for i, item in enumerate(msg['content']):
                    if isinstance(item, dict) and item.get('type') == 'image_url':
                        if isinstance(item.get('image_url'), dict) and 'url' in item['image_url']:
                            msg['content'][i]['image_url']['url'] = '[IMG_REDACTED]'
                        elif isinstance(item.get('image_url'), str):
                             msg['content'][i]['image_url'] = '[IMG_REDACTED_STR]'

    header = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - Req: {json.dumps(log_data)}"
    prompt_parts = [header, 
                    r'Rules: Use one tool per response (XML tags: attempt_completion, ask_followup_question, etc.). No conversational replies, only tool commands. Format code with proper indentation/line breaks. Disregard prior single-line/no-newline code instructions.',
                    r'Entire response in 1 markdown codeblock. No nested codeblocks.',
                    prompt_text]
    full_prompt = "\n".join(prompt_parts)
    
    llm_raw_resp = talkto("gemini", full_prompt, images)
    
    if isinstance(llm_raw_resp, str):
        logger.info(f"{selective_handler.llm_response_marker} {llm_raw_resp}") 
    else:
        logger.warning(f"LLM resp not str: {type(llm_raw_resp)}. Content: {str(llm_raw_resp)[:100]}.")

    parsed = {"raw_response_for_client": "", "has_followup": False, "has_completion": False, "has_thinking": False}
    if not isinstance(llm_raw_resp, str): return parsed

    processed = llm_raw_resp.strip()
    if processed.startswith("```") and processed.endswith("```"):
        processed = re.sub(r'^```[a-zA-Z]*\n?', '', processed)
        processed = re.sub(r'\n?```$', '', processed).strip()
    elif processed.endswith("```"): processed = processed[:-3].strip()
    
    parsed["raw_response_for_client"] = processed
    if re.search(r"<thinking>(.*?)</thinking>", processed, re.DOTALL): parsed["has_thinking"] = True
    attempt = re.search(r"<attempt_completion>(.*?)</attempt_completion>", processed, re.DOTALL)
    if attempt and re.search(r"<result>(.*?)</result>", attempt.group(1), re.DOTALL): parsed["has_completion"] = True
    if re.search(r"<ask_followup_question>(.*?)</ask_followup_question>", processed, re.DOTALL): parsed["has_followup"] = True
    return parsed

@app.route('/', methods=['GET'])
def home():
    logger.info(f"GET / from {request.remote_addr}")
    return "API Bridge: TTS Ready"

@app.route('/status', methods=['GET'])
def get_system_status():
    global is_processing_llm_request, current_system_status, system_status_message, llm_interaction_lock
    with llm_interaction_lock:
        ready_new = not is_processing_llm_request and current_system_status != "AWAITING_FOLLOWUP"
        awaiting_fup = not is_processing_llm_request and current_system_status == "AWAITING_FOLLOWUP"
        if current_system_status == "ERROR" and not is_processing_llm_request: ready_new = True
        return jsonify({
            "is_busy": is_processing_llm_request,
            "status_code": current_system_status,
            "status_message": system_status_message,
            "ready_for_new_prompt": ready_new,
            "awaiting_followup": awaiting_fup
        })

@app.route('/chat/completions', methods=['POST'])
def chat_completions():
    global is_processing_llm_request, current_system_status, system_status_message, llm_interaction_lock

    if not llm_interaction_lock.acquire(blocking=False):
        logger.warning("LLM busy. Rejecting request (503).")
        return jsonify({"error": {"message": "Server busy. Please try again."}}), 503

    is_processing_llm_request = True
    # Console message indicating processing has started for this request
    sys.stdout.write("System: Processing your request...\n")
    sys.stdout.flush()

    try:
        data = request.get_json()
        if not data or 'messages' not in data or not data['messages']:
            logger.error("Invalid request: 'messages' missing/empty.")
            # Update internal status, console message will be from error block below if this path is taken often
            with llm_interaction_lock: # Ensure consistent update
                 current_system_status = "ERROR"; system_status_message = "Invalid request structure."
            return jsonify({'error': {'message': 'Invalid request: "messages" missing/empty'}}), 400
        
        prompt = get_content_text(data['messages'][-1].get('content', ''))
        req_id, model = str(int(time.time())), data.get("model", "gpt-3.5-turbo")
        
        llm_details = handle_llm_interaction(prompt, data) # Triggers SelectiveConsolePrintHandler
        client_resp_str = llm_details["raw_response_for_client"]

        console_status_msg_after_llm = None

        with llm_interaction_lock:
            if llm_details["has_followup"]:
                current_system_status = "AWAITING_FOLLOWUP"
                system_status_message = "Awaiting your response to the follow-up question."
                console_status_msg_after_llm = f"System: {system_status_message}"
            elif llm_details["has_completion"]:
                current_system_status = "COMPLETED_NO_FOLLOWUP" 
                system_status_message = "Task completed. Ready for new input."
                console_status_msg_after_llm = f"System: {system_status_message}"
            else: 
                # LLM turn done (thinking/tools/empty), no followup/completion. System is ready.
                # No *additional specific* "System: Ready..." console message here.
                # The end of SelectiveConsolePrintHandler's output + prompt reappearing implies readiness.
                current_system_status = "IDLE"
                system_status_message = "Ready for new input." # For /status endpoint
            
        if console_status_msg_after_llm:
             sys.stdout.write(console_status_msg_after_llm + "\n")
             sys.stdout.flush()

        logger.info(f"LLM client response len: {len(client_resp_str)}")

        if data.get('stream', False):
            def gen_stream():
                yield f"data: {json.dumps({'id':f'chatcmpl-{req_id}','object':'chat.completion.chunk','created':int(time.time()),'model':model,'choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]})}\n\n"
                for line in client_resp_str.splitlines(True):
                    if not line: continue
                    yield f"data: {json.dumps({'id':f'chatcmpl-{req_id}','object':'chat.completion.chunk','created':int(time.time()),'model':model,'choices':[{'index':0,'delta':{'content':line},'finish_reason':None}]})}\n\n"
                yield f"data: {json.dumps({'id':f'chatcmpl-{req_id}','object':'chat.completion.chunk','created':int(time.time()),'model':model,'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(gen_stream(), mimetype='text/event-stream')
        
        return jsonify({
            'id':f'chatcmpl-{req_id}','object':'chat.completion','created':int(time.time()),'model':model,
            'choices':[{'index':0,'message':{'role':'assistant','content':client_resp_str},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':len(prompt.split()),'completion_tokens':len(client_resp_str.split()),'total_tokens':len(prompt.split())+len(client_resp_str.split())}
        })
    except Exception as e:
        logger.error(f"Error in /chat/completions: {e}", exc_info=True)
        with llm_interaction_lock:
            current_system_status = "ERROR"
            system_status_message = "An internal error occurred. Ready for new input."
        # This console message provides feedback on the error and that the system is ready for a retry.
        sys.stdout.write(f"System: {system_status_message}\n")
        sys.stdout.flush()
        return jsonify({'error': {'message': f"Internal server error: {e}"}}), 500
    finally:
        is_processing_llm_request = False
        llm_interaction_lock.release() 
        logger.info(f"Request done. Status: {current_system_status} ({system_status_message})")

if __name__ == '__main__':
    start_tts_service() 
    logger.info("Main: Waiting for TTS init...")
    time.sleep(3)

    sys.stdout.write("Cline x voice running\n"); sys.stdout.flush()
    if tts_engine_initialized_in_worker:
        speak_via_queue("Cline x voice running")
    else:
        logger.warning("Main: TTS not initialized, skipping startup speech.")
    
    with llm_interaction_lock:
        current_system_status = "IDLE"
        system_status_message = "Ready for new input."
    sys.stdout.write(f"System: {system_status_message}\n"); sys.stdout.flush()
    
    logger.info(f"Main: Starting server. TTS Init: {tts_engine_initialized_in_worker}. Status: {current_system_status}")
    try:
        app.run(host="0.0.0.0", port=3001, debug=False, use_reloader=False) 
    finally:
        logger.info("Main: Shutting down..."); stop_tts_service(); logger.info("Main: Shutdown complete.")