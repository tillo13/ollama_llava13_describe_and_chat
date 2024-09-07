from PIL import Image
import ollama
import atexit
import time
from ollama_utils import (
    install_and_setup_ollama,
    kill_existing_ollama_service,
    clear_gpu_memory,
    start_ollama_service_windows,
    stop_ollama_service,
    is_windows
)

def generate_description(file_path, prompt):
    """
    Generate a description for the image with the given prompt.
    """
    try:
        result = ollama.generate(
            model='llava:13b',
            prompt=prompt,
            images=[file_path],
            stream=False  # Setting stream to False to avoid receiving streaming responses
        )['response']
    except Exception as e:
        print(f"Error generating image description: {e}")
        return None
    return result

def initialize_conversation(file_path):
    """
    Initialize conversation by generating an initial description of the image.
    """
    try:
        # Open and display the image (optional, for visual confirmation)
        img = Image.open(file_path, mode='r')
        img.show()  # This will open the image using the default image viewer in your system
    except Exception as e:
        print(f"Error opening image: {e}")

    initial_prompt = "Describe this image"
    initial_response = generate_description(file_path, initial_prompt)
    if initial_response is None:
        return None, None
    message_history = [{"role": "user", "content": initial_prompt}, {"role": "assistant", "content": initial_response}]
    
    return message_history, initial_response

def reset_conversation(file_path, user_input, initial_description):
    """
    Reset the conversation context while retaining the initial image description.
    """
    prompt = user_input.replace(file_path, "").strip()
    new_response = generate_description(file_path, prompt)
    if new_response is None:
        return message_history, None

    print("Detected filename in input. Resetting conversation and re-ingesting image.")
    
    message_history = [
        {"role": "user", "content": "Describe this image"}, 
        {"role": "assistant", "content": initial_description},
        {"role": "user", "content": prompt}, 
        {"role": "assistant", "content": new_response}
    ]
    return message_history, new_response

def continue_conversation(message_history, user_input, file_path, initial_description):
    """
    Continue the conversation based on user input, resetting history if image file is mentioned.
    """
    if file_path.lower() in user_input.lower():
        message_history, response = reset_conversation(file_path, user_input, initial_description)
    else:
        message_history.append({"role": "user", "content": user_input})

        full_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in message_history])
        try:
            result = ollama.generate(
                model='llava:13b',
                prompt=full_context,
                stream=False  # Setting stream to False to avoid receiving streaming responses
            )['response']
        except Exception as e:
            print(f"Error generating response: {e}")
            return None, message_history

        message_history.append({"role": "assistant", "content": result})
        response = result
    
    return response, message_history

def main():
    # Ensure Ollama service is properly handled on exit
    atexit.register(stop_ollama_service)
    atexit.register(clear_gpu_memory)

    # Install and set up Ollama if not already done
    install_and_setup_ollama('llava:13b')

    if is_windows():
        print("Starting Ollama service on Windows...")
        if not start_ollama_service_windows():
            print("Error: Failed to start Ollama service. Exiting.")
            return
        time.sleep(10)  # Wait a bit to ensure the service is properly started

    # Define your image path
    image_to_send_in = "test.jpg"  # Replace this with your actual image path
    
    message_history, initial_response = initialize_conversation(image_to_send_in)
    if message_history is None:
        print("Failed to initialize conversation.")
        return

    initial_description = initial_response
    print("Initial description of the image:", initial_response)

    while True:
        user_input = input("Ask a clarifying question (or type 'exit' to end): ").strip()
        if user_input.lower() == 'exit':
            break
        
        response, message_history = continue_conversation(message_history, user_input, image_to_send_in, initial_description)
        if response is not None:
            print("Response:", response)

if __name__ == "__main__":
    main()