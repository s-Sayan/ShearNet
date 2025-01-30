import os

def main():
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

    print(f"Environment variable SHEARNET_DATA_PATH set to {save_path}")
    print(f"Added to {rc_file}. Please restart your terminal or run 'source {rc_file}' to apply changes.")

if __name__ == "__main__":
    main()
