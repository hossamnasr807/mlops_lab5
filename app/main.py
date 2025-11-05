import gradio as gr
import requests
from fastapi import FastAPI

# Ollama local API endpoint
LOCAL_OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# Main text generation function
def generate_text(prompt: str, model_name: str):
    if not prompt.strip():
        return "Please enter a prompt before generating."
    
    try:
        # Prepare request for Ollama local model
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(LOCAL_OLLAMA_ENDPOINT, json=data)
        response.raise_for_status()

        # Extract model response
        result = response.json().get("response", "No response received from the model.")
        return result

    except requests.exceptions.ConnectionError:
        return "Could not connect to Ollama. Make sure the Ollama server is running locally."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Dropdown for model selection
model_dropdown = gr.Dropdown(
    choices=["codellama:7b-instruct"],
    value="codellama:7b-instruct",
    label="Select Local Model"
)

# Gradio interface
gui = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            lines=4,
            label="Your Prompt",
            placeholder="Ask your AI assistant something..."
        ),
        model_dropdown
    ],
    
    outputs=gr.Textbox(
    label="AI Assistant Response",
    lines=12,           
    max_lines=20,       
    show_copy_button=True,  
    autoscroll=True     
),

    title="Local AI Assistant (Ollama + Gradio)",
    description="A fully local AI chat assistant powered by Ollama."
)

# FastAPI integration
app = FastAPI(title="Local Ollama AI Assistant")
app = gr.mount_gradio_app(app, gui, path="/")
