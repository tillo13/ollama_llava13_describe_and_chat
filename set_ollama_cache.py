import os
import shutil
import subprocess

def set_environment_variable(variable_name, variable_value, scope='user'):
    """Set an environment variable."""
    if scope == 'system':
        target = 'Machine'
    else:
        target = 'User'
    
    try:
        subprocess.run([
            "powershell", "-Command",
            f'[System.Environment]::SetEnvironmentVariable("{variable_name}", "{variable_value}", [System.EnvironmentVariableTarget]::{target})'
        ], check=True)
        print(f"Environment variable {variable_name} set to {variable_value} for {target.lower()}.")
    except subprocess.CalledProcessError as e:
        print(f"Error setting environment variable: {e}")

def move_and_link_directory(original_path, new_path):
    """Move a directory and create a symbolic link."""
    try:
        # Move the models directory to the new location
        shutil.move(original_path, new_path)
        print(f"Moved models directory from {original_path} to {new_path}.")

        # Create the symbolic link
        os.symlink(new_path, original_path, target_is_directory=True)
        print(f"Created symbolic link from {original_path} to {new_path}.")
    except Exception as e:
        print(f"Error moving and linking directory: {e}")

def main():
    original_path = os.path.expanduser("~/.ollama/models")
    new_path = "D:/ollama_models"

    # Ensure new path exists
    os.makedirs(new_path, exist_ok=True)

    move_and_link_directory(original_path, new_path)
    set_environment_variable("OLLAMA_MODELS", new_path, scope='user')

if __name__ == "__main__":
    main()