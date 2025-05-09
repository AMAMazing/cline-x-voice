import pyttsx3
from threading import Thread, Event
import time
import logging
import sys
import queue # For inter-thread communication

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("tts_single_thread_test.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- TTS Globals ---
tts_queue = queue.Queue()
tts_thread_stop_event = Event() # To signal the TTS thread to stop
tts_engine_global = None # Keep a reference if needed for shutdown, though init/use is in thread
tts_initialized_in_thread = False


def tts_worker():
    global tts_engine_global, tts_initialized_in_thread
    engine = None
    init_attempts_thread = 0

    logger.info("TTS WORKER: Thread started.")
    try:
        init_attempts_thread += 1
        logger.info(f"TTS WORKER: Attempting to initialize TTS engine (Thread Attempt: {init_attempts_thread})...")
        if sys.platform == "win32":
            engine = pyttsx3.init(driverName='sapi5')
        else:
            engine = pyttsx3.init()

        if engine:
            tts_engine_global = engine # Store reference
            tts_initialized_in_thread = True
            logger.info(f"TTS WORKER: pyttsx3.init() successful. Engine object: {id(engine)}")
            
            # Optional: Speak an init message from within the worker
            engine.say("TTS worker thread initialized.")
            engine.runAndWait()
            logger.info("TTS WORKER: Spoke worker initialization message.")
        else:
            logger.error("TTS WORKER: pyttsx3.init() returned None in worker thread.")
            tts_initialized_in_thread = False
            return # Cannot proceed without engine

        while not tts_thread_stop_event.is_set():
            try:
                # Wait for a short time for an item, then check stop_event again
                text_to_say = tts_queue.get(timeout=0.1) 
                if text_to_say is None: # Sentinel value to indicate shutdown
                    logger.info("TTS WORKER: Received None (sentinel), preparing to exit.")
                    tts_thread_stop_event.set() # Ensure loop terminates
                    break 

                logger.info(f"TTS WORKER: Got from queue: '{text_to_say}'. Attempting to speak.")
                engine.say(str(text_to_say))
                engine.runAndWait()
                logger.info(f"TTS WORKER: Finished speaking: '{text_to_say}'.")
                tts_queue.task_done() # Signal that the item from the queue is processed
            except queue.Empty:
                # Timeout occurred, just loop again to check tts_thread_stop_event
                continue
            except Exception as e:
                logger.error(f"TTS WORKER: Error during speech: {e}", exc_info=True)
                # Potentially try to re-initialize or just log and continue/break
                # For now, we'll let it continue trying for next item unless stop_event is set
        
    except Exception as e_init:
        logger.error(f"TTS WORKER: Major error during initialization or loop: {e_init}", exc_info=True)
        tts_initialized_in_thread = False
    finally:
        if engine and hasattr(engine, 'stop'):
            logger.info("TTS WORKER: Attempting to stop engine in finally block.")
            # engine.stop() # Sometimes causes issues on SAPI5, runAndWait should handle event loop.
        logger.info("TTS WORKER: Thread finished.")

if __name__ == "__main__":
    logger.info("--- Starting TTS Single-Thread Queue Test ---")

    # Start the dedicated TTS worker thread
    speech_thread = Thread(target=tts_worker, daemon=True)
    speech_thread.start()
    logger.info("Main thread: TTS worker thread started.")

    # Wait a moment for the TTS worker to initialize
    time.sleep(3) # Give it time to init and speak its own init message

    if not tts_initialized_in_thread:
        logger.error("Main thread: TTS worker thread failed to initialize. Aborting test.")
    else:
        logger.info("\n--- Test Call 1 (via Queue) ---")
        tts_queue.put("This is the first message via queue.")
        
        logger.info("Main thread: Waiting for 5 seconds...")
        time.sleep(5)

        logger.info("\n--- Test Call 2 (via Queue) ---")
        tts_queue.put("This is the second message, also via queue, after a pause.")

        logger.info("Main thread: Waiting for all items in queue to be processed (max 10s)...")
        try:
            tts_queue.join(timeout=10) # Wait for task_done() to be called for all items
            if tts_queue.empty():
                logger.info("Main thread: TTS queue is empty, all tasks likely processed.")
            else:
                logger.warning("Main thread: TTS queue join timed out or queue not empty.")
        except Exception as e_join:
            logger.error(f"Main thread: Error during queue.join(): {e_join}")


    logger.info("Main thread: Signaling TTS worker to stop.")
    tts_queue.put(None) # Send sentinel to stop the worker
    
    logger.info("Main thread: Waiting for TTS worker thread to join (max 5s)...")
    speech_thread.join(timeout=5) # Wait for the thread to finish
    if speech_thread.is_alive():
        logger.warning("Main thread: TTS worker thread did not stop in time.")
    else:
        logger.info("Main thread: TTS worker thread joined successfully.")

    logger.info("--- TTS Single-Thread Queue Test Finished ---")