# Audio Analysis Application

## Overview

This Audio Analysis Application is a powerful tool that combines speech recognition, natural language processing (NLP), and large language model (LLM) capabilities to transcribe audio, analyze the content, and answer questions about the transcribed text. The application uses state-of-the-art technologies including Whisper for transcription, NLTK for text processing, and Claude 3.5 for question answering.

## Features

- Audio transcription using OpenAI's Whisper model
- Text processing and analysis using NLTK
- Named entity recognition
- Word frequency analysis
- Sentence complexity calculation
- Question answering using Claude 3.5 (Anthropic's LLM)
- CUDA support for GPU acceleration (when available)

## Requirements

- Python 3.7+
- FFmpeg
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/audio-analysis-app.git
   cd audio-analysis-app
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

5. Install FFmpeg:
   - On Ubuntu or Debian: `sudo apt update && sudo apt install ffmpeg`
   - On macOS with Homebrew: `brew install ffmpeg`
   - On Windows, download from the official FFmpeg website and add it to your system PATH.

6. Create a `.env` file in the project directory and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

1. Ensure your virtual environment is activated.

2. Run the script:
   ```
   python audio_analysis_app.py
   ```

3. When prompted, enter the path to your audio file.

4. The application will transcribe the audio and perform initial analysis.

5. You can then ask questions about the audio content. The application will use Claude 3.5 to provide answers based on the transcribed and analyzed text.

6. Type 'quit' to exit the application.

7. When you're done, deactivate the virtual environment:
   ```
   deactivate
   ```

## How It Works

1. **Audio Transcription**: The application uses Whisper to convert the audio file into text. If a CUDA-compatible GPU is available, it will be used for faster processing.

2. **Text Processing**: The transcribed text is processed using NLTK. This includes tokenization, part-of-speech tagging, named entity recognition, and various text analyses.

3. **Analysis**: The application performs several analyses on the processed text, including:
   - Identifying named entities
   - Calculating word frequencies
   - Determining the most common words and phrases
   - Estimating sentence complexity

4. **Question Answering**: Users can ask questions about the audio content. The application sends the processed text along with the user's question to Claude 3.5, which provides an answer based on the context.

## Customization

- You can modify the Whisper model size in the `transcribe_audio` function for different accuracy/speed trade-offs.
- Adjust the `max_tokens` parameter in the `analyze_with_llm` function to control the length of Claude's responses.
- Implement the `detect_inference` function to add custom logic for detecting inferences or indirect references in the text.

## Troubleshooting

- If you encounter CUDA-related errors, ensure you have the latest NVIDIA drivers installed.
- If the transcription fails, check that FFmpeg is properly installed and accessible from the command line.
- For API-related issues, verify that your Anthropic API key is correct and has the necessary permissions.

## Contributing

Contributions to improve the application are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new virtual environment and install the requirements as described in the Installation section.
3. Make your changes and test them thoroughly.
4. Update the requirements.txt file if you've added or updated any dependencies:
   ```
   pip freeze > requirements.txt
   ```
5. Submit a pull request with your changes.

By contributing to this project, you agree to license your contributions under the Apache License 2.0.

[License, Copyright Notice, and Acknowledgments sections remain unchanged]

## Copyright Notice

Copyright [2024] [Raul Pineda]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Acknowledgments

- OpenAI for the Whisper model
- NLTK contributors
- Anthropic for the Claude API

These acknowledgments do not imply endorsement of this project by the acknowledged parties.

For any questions or issues, please open an issue on the GitHub repository.