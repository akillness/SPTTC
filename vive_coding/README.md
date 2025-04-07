# Solar Chatbot

This is a simple chatbot implementation using the TinySolar-248m-4k-py-instruct model from Upstage.

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- At least 4GB of VRAM

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the chatbot:
```bash
python chatbot.py
```

- Type your messages and press Enter to chat with the bot
- Type 'quit' to exit the conversation

## Features

- Uses 8-bit quantization for efficient memory usage
- Supports Korean and English conversations
- Simple command-line interface
- Error handling for robustness

## Notes

- The model is optimized for Python-related tasks but can handle general conversations
- Response generation might take a few seconds depending on your hardware
- The model's responses are generated based on its training data and may not always be perfect 