import ollama
from PIL import Image
import os
import json
import time
import atexit
from ollama_utils import (
    install_and_setup_ollama,
    kill_existing_ollama_service,
    clear_gpu_memory,
    start_ollama_service_windows,
    stop_ollama_service,
    is_windows
)

IMAGE_FILE_PATH = "test3.png"
TEXT_BASED_OLLAMA_MODEL = 'wizard-vicuna-uncensored'
TEXT_BASED_OLLAMA_MODEL = 'llama3.1'
HISTORY_FILE_PATH = "history.json"  # Path to save the conversation history
SYSTEM_PROMPT = "you are a master at the online gaming community of roblox"

def delete_history_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted history file: {file_path}")

def save_conversation_history(file_path, message_history):
    with open(file_path, 'w') as file:
        json.dump(message_history, file, indent=2)

def load_conversation_history(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

def generate_image_description(file_path, prompt, model='llava:13b'):
    """
    Generate a description for the image using the given prompt.
    """
    print(f"Generating image description with model {model} for {file_path}")
    try:
        result = ollama.generate(
            model=model,
            prompt=prompt,
            images=[file_path],
            stream=False  # Setting stream to False to avoid receiving streaming responses
        )['response']
    except ollama._types.ResponseError as e:
        error_message = str(e)  # Capture the error message
        if "model" in error_message.lower() and "not found" in error_message.lower():
            print("Model not found. Trying to pull the model...")
            try:
                install_and_setup_ollama(model)
                result = ollama.generate(
                    model=model,
                    prompt=prompt,
                    images=[file_path],
                    stream=False
                )['response']
            except Exception as inner_err:
                print(f"Failed to pull the model or other error occurred: {inner_err}")
                raise
        else:
            print(f"Response error encountered: {error_message}")
            raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
    print(f"Image description generated: {result}")
    return result

def initialize_conversation(file_path):
    """
    Initialize conversation by generating an initial description of the image.
    """
    print(f"Initializing conversation with image: {file_path}")
    img = Image.open(file_path, mode='r')

    # Include the system prompt to set the behavior as a pirate
    initial_prompt = f"{SYSTEM_PROMPT}\n\nDescribe this image"
    initial_response = generate_image_description(file_path, initial_prompt)
    message_history = [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": "Describe this image"},
                       {"role": "assistant", "content": initial_response}]
    
    # Save the initial conversation to the file
    save_conversation_history(HISTORY_FILE_PATH, message_history)

    return message_history, initial_response

def reset_conversation(new_image_file_path, user_input, initial_description):
    """
    Reset the conversation context while retaining the initial image description for a new image.
    """
    prompt = user_input.replace(new_image_file_path, "").strip()
    initial_description = generate_image_description(new_image_file_path, f"{SYSTEM_PROMPT}\n\nDescribe this image in intricate detail.")
    new_response = generate_image_description(new_image_file_path, f"{SYSTEM_PROMPT}\n\n{prompt}")
    print("Detected image file in input. Resetting conversation and re-ingesting new image.")
    
    message_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Describe this image"},
        {"role": "assistant", "content": initial_description},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": new_response}
    ]
    
    # Save the reset conversation to the history file
    save_conversation_history(HISTORY_FILE_PATH, message_history)

    return message_history, new_response, initial_description

def generate_text_response(model_name, prompt):
    """
    Generate text-based conversation response using the given model.
    """
    print(f"Generating text response with model {model_name}")
    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        stream=False  # Setting stream to False to avoid receiving streaming responses
    )['response']
    print(f"Text response generated: {response}")
    return response

def continue_conversation(message_history, user_input, initial_description):
    """
    Continue the conversation based on user input, resetting history if new image file is mentioned.
    """
    print(f"Continuing conversation with input: {user_input}")

    new_image_file_path = None
    if ".jpg" in user_input.lower() or ".png" in user_input.lower():
        words = user_input.split()
        for word in words:
            if word.lower().endswith(('.jpg', '.png')):
                if os.path.exists(word):
                    new_image_file_path = word
                    break

    if new_image_file_path:
        message_history, response, initial_description = reset_conversation(new_image_file_path, user_input, initial_description)
    else:
        message_history.append({"role": "user", "content": user_input})

        # Use text-based model with the latest conversational context
        full_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in message_history])
        response = generate_text_response(TEXT_BASED_OLLAMA_MODEL, full_context)

        message_history.append({"role": "assistant", "content": response})
    
        # Save the updated conversation history to the file
        save_conversation_history(HISTORY_FILE_PATH, message_history)

    return response, message_history, initial_description

def print_colored(text, color="yellow"):
    """
    Simple method to print colored text in the terminal.
    """
    colors = {
        "yellow": "\033[93m",
        "reset": "\033[0m"
    }
    print(colors.get(color, colors["reset"]) + text + colors["reset"])

if __name__ == "__main__":
    atexit.register(stop_ollama_service)
    atexit.register(clear_gpu_memory)
    
    # Delete history file at the start
    delete_history_file(HISTORY_FILE_PATH)
    
    print("Killing existing Ollama service...")
    kill_existing_ollama_service()
    clear_gpu_memory()
    
    print("Installing and setting up Ollama...")
    install_and_setup_ollama(TEXT_BASED_OLLAMA_MODEL)
    
    if is_windows():
        print("Starting Ollama service on Windows...")
        start_ollama_service_windows()
        time.sleep(10)
    
    # Initialize a new conversation
    try:
        message_history, initial_response = initialize_conversation(IMAGE_FILE_PATH)
        initial_description = initial_response
        print("Initial description of the image:", initial_response)
    except Exception as e:
        print(f"Failed to initialize the conversation: {e}")
        exit(1)
    
    # Interactive loop for conversation
    while True:
        user_input = input("Ask a clarifying question (or type 'exit' to end): ").strip()
        if user_input.lower() == 'exit':
            break
        
        try:
            response, message_history, initial_description = continue_conversation(message_history, user_input, initial_description)
        except Exception as e:
            print(f"Failed to continue the conversation: {e}")
            break
        
        print("\n====================\nYou:\n====================")
        print(user_input)
        print("\n====================\nBot:\n====================")
        print_colored(response, "yellow")

    print("Stopping Ollama service and clearing GPU memory...")
    stop_ollama_service()
    clear_gpu_memory()