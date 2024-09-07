from PIL import Image
import ollama
import atexit
import time  # Import the missing time module
from ollama_utils import (
    install_and_setup_ollama,
    kill_existing_ollama_service,
    clear_gpu_memory,
    start_ollama_service_windows,
    stop_ollama_service,
    is_windows
)

def generate_image_description(file_path, instruction):
    """
    Generate a description for the image using the provided instruction.
    """
    try:
        # Use ollama's generate function to send the image and instruction to the LLaVa model
        result = ollama.generate(
            model='llava:13b',
            prompt=instruction,
            images=[file_path],
            stream=False  # Setting stream to False to avoid receiving streaming responses
        )['response']
    except Exception as e:
        print(f"Error generating image description: {e}")
        return None

    try:
        # Open and display the image (optional, for visual confirmation)
        img = Image.open(file_path, mode='r')
        img.show()  # This will open the image using the default image viewer in your system
    except Exception as e:
        print(f"Error opening image: {e}")

    return result

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
    image_to_send_in = "andy.jpg"  # Replace this with your actual image path

    # List of instructions
    instructions = [
        "Describe this image",
        "Reply yes or no. Is this man wearing a hat",
        "What color are this man's eyes",
        "Estimate this man's age within 5 years, you must pick a number",
        "Rate this man's looks from 1-10 where 10 is the most attractive human on the planet and 1 is ugly. Use any opinions you want, but you must pick a number, respond with only a number"
    ]

    while True:
        # Allow user to select an instruction
        print("\nAvailable Instructions:")
        for idx, instr in enumerate(instructions, 1):
            print(f"{idx}: {instr}")

        try:
            choice = input("Select an instruction by number (or type 'exit' to leave): ").strip()
            if choice.lower() == 'exit':
                break
            choice = int(choice)
            if 1 <= choice <= len(instructions):
                instruction = instructions[choice - 1]
            else:
                print("Invalid choice. Please select a valid instruction number.")
                continue
        except ValueError:
            print("Invalid input. Please enter a number or 'exit'.")
            continue

        # Generate and print the image description
        description = generate_image_description(image_to_send_in, instruction)
        if description is not None:
            print("\nDescription of the image:", description)

if __name__ == "__main__":
    main()