"""Post-installation helper for ShearNet.

Ensures the directory where ShearNet stores trained models and plots exists, and
tells the user how to point ShearNet at a different location via the
``SHEARNET_DATA_PATH`` environment variable.

By default this script does NOT modify your shell configuration files. Pass
``--write-shell-config`` to append the export line to your shell profile
(idempotently). ShearNet works without any of this — when ``SHEARNET_DATA_PATH``
is unset it simply uses the current working directory.

Usage:
    python scripts/post_installation.py [--data-path PATH] [--write-shell-config]
"""

import argparse
import os


def resolve_data_path(explicit=None):
    """Return the chosen data path: ``--data-path`` > env var > current directory."""
    if explicit:
        return os.path.abspath(explicit)
    return os.environ.get("SHEARNET_DATA_PATH", os.path.abspath("."))


def shell_rc_file():
    """Best-effort path to the user's shell startup file."""
    shell = os.getenv("SHELL", "/bin/bash")
    home = os.path.expanduser("~")
    if "zsh" in shell:
        return os.path.join(home, ".zshrc")
    if "bash" in shell:
        bashrc = os.path.join(home, ".bashrc")
        return bashrc if os.path.exists(bashrc) else os.path.join(home, ".bash_profile")
    return os.path.join(home, ".profile")


def write_shell_config(data_path):
    """Append the SHEARNET_DATA_PATH export to the shell profile, if not already present."""
    rc_file = shell_rc_file()
    export_line = f'export SHEARNET_DATA_PATH="{data_path}"'
    if os.path.exists(rc_file):
        with open(rc_file, "r") as f:
            if export_line in f.read():
                print(f"SHEARNET_DATA_PATH already set in {rc_file}; leaving it unchanged.")
                return
    with open(rc_file, "a") as f:
        f.write(f"\n{export_line}\n")
    print(f"Added SHEARNET_DATA_PATH to {rc_file}.")
    print(f"Run 'source {rc_file}' or restart your terminal to apply it.")


def main():
    parser = argparse.ArgumentParser(description="ShearNet post-installation setup.")
    parser.add_argument("--data-path", default=None,
                        help="Directory for trained models and plots "
                             "(default: $SHEARNET_DATA_PATH or the current directory).")
    parser.add_argument("--write-shell-config", action="store_true",
                        help="Persist SHEARNET_DATA_PATH by appending an export line "
                             "to your shell profile (off by default).")
    args = parser.parse_args()

    data_path = resolve_data_path(args.data_path)
    os.makedirs(data_path, exist_ok=True)

    print(f"ShearNet will store models and plots under: {data_path}")
    if args.write_shell_config:
        write_shell_config(data_path)
    else:
        print("To use a different location, set the SHEARNET_DATA_PATH environment "
              "variable, e.g.:")
        print(f'    export SHEARNET_DATA_PATH="{data_path}"')
        print("Add that line to your shell profile to make it permanent, or re-run "
              "this script with --write-shell-config to do it automatically.")


if __name__ == "__main__":
    main()
