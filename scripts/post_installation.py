import os

def main():
    # Check if SHEARNET_DATA_PATH already exists
    existing_path = os.environ.get('SHEARNET_DATA_PATH')
    
    if existing_path:
        print(f"SHEARNET_DATA_PATH is already set to: {existing_path}")
        response = input("Do you want to change it? [y/N]: ").strip().lower()
        
        if response not in ['y', 'yes']:
            print(f"Keeping existing SHEARNET_DATA_PATH: {existing_path}")
            return
    
    # Resolve the absolute path of the current directory
    default_path = os.path.abspath('.')
    save_path = input(f"Enter the directory to save trained models and plots (default: {default_path}): ").strip()
    
    # Use the default path if the user doesn't provide input
    if not save_path:
        save_path = default_path
    
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Determine the shell configuration file
    shell = os.getenv('SHELL', '/bin/bash')
    home_dir = os.path.expanduser('~')
    if 'zsh' in shell:
        rc_file = os.path.join(home_dir, '.zshrc')
    elif 'bash' in shell:
        # Prefer .bashrc if it exists, otherwise use .bash_profile
        if os.path.exists(os.path.join(home_dir, '.bashrc')):
            rc_file = os.path.join(home_dir, '.bashrc')
        else:
            rc_file = os.path.join(home_dir, '.bash_profile')
    else:
        rc_file = os.path.join(home_dir, '.profile')  # Fallback for other shells

    # Add the environment variable to the shell configuration file
    env_variable = f'\nexport SHEARNET_DATA_PATH="{save_path}"\n'
    with open(rc_file, 'a') as f:
        f.write(env_variable)

    # Set the environment variable for the current session
    os.environ['SHEARNET_DATA_PATH'] = save_path

    print(f"Environment variable SHEARNET_DATA_PATH set to {save_path}")
    print(f"Added to {rc_file}. Please restart your terminal or run 'source {rc_file}' to apply changes.")

if __name__ == "__main__":
    main()