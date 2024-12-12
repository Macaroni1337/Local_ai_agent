import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import pyttsx3  # For text-to-speech
import speech_recognition as sr  # For speech-to-text
import streamlit as st  # For web interface
import os
import json
import requests  # For API integrations

class LocalAIAgent:
    def __init__(self, model_path: str, context_file: str = "context.json"):
        """
        Initialize the AI agent with the LLaMA model, tokenizer, and context persistence.

        :param model_path: Path to the local LLaMA model files.
        :param context_file: File path for saving/loading conversation context.
        """
        print("Loading model and tokenizer...")
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("Model loaded successfully!")
        self.context_file = context_file
        self.context = self.load_context()
        self.tts_engine = pyttsx3.init()

    def load_context(self):
        """Load conversation context from a file."""
        if os.path.exists(self.context_file):
            with open(self.context_file, 'r') as f:
                return json.load(f)
        return []

    def save_context(self):
        """Save conversation context to a file."""
        with open(self.context_file, 'w') as f:
            json.dump(self.context[-10:], f)  # Save only the last 10 exchanges

    def generate_response(self, prompt: str, max_length: int = 200):
        """
        Generate a response based on the given prompt and conversation history.

        :param prompt: User input or query.
        :param max_length: Maximum length of the response.
        :return: The AI's response as a string.
        """
        self.context.append(f"You: {prompt}")
        full_context = "\n".join(self.context[-10:])  # Use the last 10 exchanges for context

        inputs = self.tokenizer(full_context, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs["input_ids"], 
            max_length=max_length, 
            temperature=0.7, 
            top_k=50, 
            top_p=0.95,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.context.append(f"AI: {response}")
        self.save_context()
        return response

    def text_to_speech(self, text: str):
        """Convert text to speech."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def speech_to_text(self):
        """Convert speech to text."""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = recognizer.listen(source, timeout=5)
                return recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "Sorry, I didn't catch that."
            except sr.WaitTimeoutError:
                return "No speech detected."

    def perform_task(self, task: str):
        """Perform specific tasks like summarization, email drafting, or API calls."""
        if task.startswith("summarize "):
            _, file_path = task.split("summarize ", 1)
            return self.summarize_file(file_path.strip())
        elif task.startswith("draft email "):
            content = task.split("draft email ", 1)[1]
            return f"Drafting an email based on your input: {content}"
        elif task.startswith("get weather "):
            location = task.split("get weather ", 1)[1]
            return self.get_weather(location)
        else:
            return "Task not recognized. Please try again."

    def summarize_file(self, file_path: str):
        """Summarize the contents of a text file."""
        if not os.path.exists(file_path):
            return f"File '{file_path}' not found."
        with open(file_path, 'r') as file:
            content = file.read()
            return f"Summary: {content[:200]}..."  # Simplified summarization

    def get_weather(self, location: str):
        """Fetch current weather for a given location using a public API."""
        api_key = "your_openweather_api_key"  # Replace with your API key
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return f"Weather in {location}: {data['weather'][0]['description']}, {data['main']['temp']}°C"
        return "Unable to fetch weather data. Please check the location."

# Streamlit Web Interface
def web_interface(agent: LocalAIAgent):
    st.set_page_config(page_title="Local AI Agent", layout="wide")
    st.title("Local AI Agent")

    st.sidebar.title("Options")
    use_voice = st.sidebar.checkbox("Use Voice Input")
    enable_tts = st.sidebar.checkbox("Enable Text-to-Speech")

    user_input = st.text_input("Your Query", placeholder="Type something...")

    if use_voice:
        st.write("Listening for your voice input...")
        user_input = agent.speech_to_text()
        st.write(f"Recognized: {user_input}")

    if st.button("Submit"):
        if user_input:
            if user_input.startswith("task: "):
                task_output = agent.perform_task(user_input[6:])
                st.text_area("Task Output", value=task_output, height=200)
                if enable_tts:
                    agent.text_to_speech(task_output)
            else:
                response = agent.generate_response(user_input)
                st.text_area("AI Response", value=response, height=200)
                if enable_tts:
                    agent.text_to_speech(response)
        else:
            st.warning("Please enter a query or use voice input.")

def main():
    # Path to your local LLaMA model files
    model_path = "path/to/llama/model"

    # Initialize the AI agent
    agent = LocalAIAgent(model_path)

    # Choose interaction mode
    mode = input("Choose mode: [console/web] ").strip().lower()
    if mode == "console":
        print("Welcome to your local AI agent!")
        print("Type 'exit' to quit. Use 'voice' for voice input.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            elif user_input.lower() == "voice":
                user_input = agent.speech_to_text()
                print(f"You (via voice): {user_input}")

            if user_input:
                if user_input.startswith("task: "):
                    print(agent.perform_task(user_input[6:]))
                else:
                    response = agent.generate_response(user_input)
                    print(f"AI: {response}")
                    agent.text_to_speech(response)
    elif mode == "web":
        web_interface(agent)
    else:
        print("Invalid mode. Please restart and choose 'console' or 'web'.")

if __name__ == "__main__":
    main()
