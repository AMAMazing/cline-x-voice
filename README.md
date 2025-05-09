# Cline X Voice (Local)

A Python-based API bridge that enables Cline (VS Code Extension) to interact with LLMs web interface, providing OpenAI-compatible API endpoints and Text-to-Speech (TTS) feedback for seamless integration.

## Overview

This project creates a Flask server that acts as a middleware between Cline and an LLM webchat interface, translating API requests into web interactions and providing voice feedback for certain events. It simulates an OpenAI-compatible API endpoint, allowing Cline to use the web interface as if it were communicating with the OpenAI API, with the added benefit of audible responses for completions and questions.

## Features

- OpenAI-compatible API endpoints
- **Text-to-Speech (TTS) for attempt completions and follow-up questions**
- Automated browser interaction with LLM logged in
- Request rate limiting and management
- Clipboard-based data transfer
- Streaming response support
- Comprehensive error handling and logging
- Support for various content formats

## Prerequisites

- Python 3.6+
- Windows OS (due to win32clipboard and pyttsx3 SAPI5 driver dependency for optimal performance)
- Chrome/Firefox browser installed
- Active LLM account logged in

Required Python packages:
```
flask
pywin32
pyautogui
optimisewait
pyttsx3
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AMAMazing/cline-x-voice.git 
```
(Note: The user provided a generic cline-x repo link. Assuming a new repo for cline-x-voice or user will adapt.)

2. Install dependencies:
```bash
pip install flask pywin32 pyautogui optimisewait pyttsx3
```

3. Set up the image directory structure (required for GUI automation):
```
images/
├── approve.png
├── copy.png
├── error.png
├── instructions.png
├── likedislike.png
├── proceed.png
├── proceed2.png
├── resume.png
├── run.png
├── runcommand.png
├── save.png
├── startnewtask.png
├── typesmthn.png
└── alt1440/
    ├── claudenew.png
    ├── copy.png
    └── submit.png
```
(Ensure these images are relevant or update as needed for the target LLM interface if it's different from the original `cline-x` target.)

## Configuration

1. Update the image paths in the relevant GUI automation script (e.g., a file like `claude.py` or `talktollm.py` if GUI automation is still used for the LLM interaction part, or in `main.py` if paths are there):
```python
# Example if using optimisewait directly in main.py or a similar module
# from optimisewait import set_autopath, set_altpath
# set_autopath(r"path/to/your/images")
# set_altpath(r"path/to/your/images/alt1440")
```
(The provided `main.py` for cline-x-voice uses `set_autopath(r"D:\\cline-x-claudeweb\\images")`. This should be configurable by the user.)

2. Adjust the `MIN_REQUEST_INTERVAL` in `main.py` (default: 5 seconds) if needed to match your rate limiting requirements.

3. TTS Configuration (Optional):
   - The TTS engine (`pyttsx3`) attempts to initialize with SAPI5 on Windows. Other drivers might be used on other OS, but Windows is preferred.
   - Voice, rate, and volume can be configured programmatically if `pyttsx3` engine properties are exposed or modified in the `tts_worker_function` in `main.py`.

## Usage

1. Start the server:
```bash
python main.py
```

2. Configure Cline to use the local API endpoint:
   - Open Cline settings in VS Code
   - Select "OpenAI Compatible" as the API provider
   - Set Base URL to: `http://localhost:3001`
   - Set API Key to any non-empty value (e.g., "any-value")
   - Set Model ID to "gpt-3.5-turbo" (or the model you intend to proxy)

The server will now:
1. Receive API requests from Cline.
2. Interact with the configured LLM (e.g., via `talktollm.py` which might use browser automation or direct API calls to another service).
3. Retrieve the response from the LLM.
4. If the response is an attempt_completion or ask_followup_question, the relevant text will be spoken via TTS.
5. Return formatted response to Cline.

## Technical Details

### API Endpoint

- POST `/chat/completions`: Main endpoint for chat completions
- GET `/`: Health check endpoint

### Key Components

- **Flask Server**: Handles HTTP requests and provides API endpoints.
- **Text-to-Speech (TTS) Module**: Uses `pyttsx3` in a separate thread to provide non-blocking audio feedback for completions and questions.
- **LLM Interaction (via `talktollm.py` or similar)**: Handles communication with the target Large Language Model. This might involve browser automation (PyAutoGUI, optimisewait) or direct API calls.
- **Clipboard Management**: May be used for data transfer if browser automation is involved.
- **Response Processing**: Cleans and formats LLM responses to match OpenAI API structure and extracts text for TTS.

### Rate Limiting

The server implements a simple rate limiting mechanism:
- Minimum 5-second interval between requests (configurable).
- Automatic request queueing if interval not met.

### Error Handling

- Comprehensive logging system (`app_full.log`).
- Graceful error handling for API requests.
- TTS initialization and runtime error logging.

## Limitations

- Windows-only support strongly recommended for full functionality (due to `win32clipboard` and `pyttsx3` SAPI5 preferences).
- If using GUI automation for LLM interaction:
    - Requires active browser window.
    - Depends on GUI automation (sensitive to UI changes of the target LLM web interface).
    - Requires logged-in web session for the LLM.
- Rate-limited by design (and potentially by the target LLM).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](LICENSE) (Assuming the license remains MIT, check your LICENSE file)

## Disclaimer

This project is not officially affiliated with any LLM provider (e.g., Anthropic, OpenAI) or Cline. Use at your own discretion and in accordance with respective terms of service.
