Local AI Agent



Overview

The Local AI Agent is a personal AI assistant designed to run on your local machine, leveraging the LLaMA language model. It offers a console and web-based interface, supports context-aware conversations, integrates with APIs for task automation, and allows text-to-speech and speech-to-text capabilities.


Features

Context-Aware Conversations: Keeps track of the last 10 exchanges for continuity.



Task Automation:

File summarization

Email drafting

Fetching weather data via OpenWeather API

Interactive Web Interface: Built with Streamlit for a user-friendly experience.

Speech-to-Text: Use voice input to interact with the agent.

Text-to-Speech: Converts AI responses to audio.



Requirements

Python 3.8+

Dependencies:

torch

transformers

pyttsx3

SpeechRecognition

streamlit

requests


Installation

Clone the repository: https://github.com/Macaroni1337/Local_ai_agent.git

git clone 


Install the required Python packages:

pip install torch transformers pyttsx3 SpeechRecognition streamlit requests

Download the LLaMA model and place it in the appropriate directory.



Setup

Update the model_path in advanced_ai_agent.py to point to the location of your LLaMA model files.

Replace your_openweather_api_key with your OpenWeather API key to enable weather fetching.

Usage

Console Mode

Run the script in console mode:

python advanced_ai_agent.py

Follow the prompts to interact with the AI agent via text or voice.

To exit, type exit.



Web Interface

Run the script in web mode:

streamlit run advanced_ai_agent.py

Open the provided URL in your browser (e.g., http://localhost:8501).

Interact with the AI using the web interface.


Commands

Contextual Queries: Enter a query or question for the AI.

Task Automation:

task: summarize <file_path>: Summarize the contents of a text file.

task: draft email <content>: Draft an email based on the provided content.

task: get weather <location>: Fetch the current weather for the specified location.



Customization

Modify the perform_task method to add new tasks.

Adjust the generate_response parameters (e.g., temperature, top_k, top_p) to fine-tune the AI's behavior.





License

This is open source, feel free to use!

Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

Contact

For issues or feature requests, please reachout to me via rafe.fredericks@spectrumstream.media (make sure the subject line mentions AI)