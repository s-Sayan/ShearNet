"""
ShearNet output management system for notebook experiments.

This module provides centralized output handling for plots and terminal output
specifically designed for ShearNet notebooks.
"""

import os
import matplotlib.pyplot as plt
from typing import Optional
from contextlib import contextmanager
from datetime import datetime
import numpy as np

class ShearNetOutputManager:
    """Manages output for ShearNet notebook experiments."""
    
    def __init__(self, debug: bool = True) -> None:
        """Initialize output manager for ShearNet notebooks."""
        self.debug = debug
        
        # Find the output directory
        self.base_dir = self._find_notebooks_out_dir()
        self.output_file = os.path.join(self.base_dir, "out.md")
        
        if self.debug:
            print(f"DEBUG: Attempting to create directory: {self.base_dir}")
        
        # Create directory if it doesn't exist
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            if self.debug:
                print(f"DEBUG: Directory created/exists: {self.base_dir}")
                print(f"DEBUG: Directory is writable: {os.access(self.base_dir, os.W_OK)}")
        except Exception as e:
            print(f"ERROR: Failed to create directory {self.base_dir}: {e}")
            # Fallback to current directory
            self.base_dir = os.path.join(os.getcwd(), "notebook_out")
            os.makedirs(self.base_dir, exist_ok=True)
            self.output_file = os.path.join(self.base_dir, "out.md")
            print(f"FALLBACK: Using directory: {self.base_dir}")
        
        # Initialize output file
        self._initialize_output_file()
        
        print(f"ShearNet Output Manager initialized:")
        print(f"  Output directory: {self.base_dir}")
        print(f"  Output file: {self.output_file}")
        print(f"  Directory exists: {os.path.exists(self.base_dir)}")
        print(f"  Can write to directory: {os.access(self.base_dir, os.W_OK)}")
    
    def _find_notebooks_out_dir(self) -> str:
        """Find the notebooks/out directory using a simpler approach."""
        # Get current working directory
        cwd = os.getcwd()
        
        if self.debug:
            print(f"DEBUG: Current working directory: {cwd}")
        
        # Strategy 1: Check if we're already in a notebooks directory
        if 'notebooks' in cwd.lower():
            if self.debug:
                print("DEBUG: Found 'notebooks' in current path")
            # We're in or under a notebooks directory
            if cwd.endswith('notebooks'):
                return os.path.join(cwd, "out")
            else:
                # We're in a subdirectory of notebooks, go up to find notebooks
                parts = cwd.split(os.sep)
                try:
                    notebooks_idx = [p.lower() for p in parts].index('notebooks')
                    notebooks_path = os.sep.join(parts[:notebooks_idx+1])
                    return os.path.join(notebooks_path, "out")
                except ValueError:
                    pass
        
        # Strategy 2: Look for notebooks directory in current location or parents
        current_path = cwd
        for _ in range(5):  # Search up to 5 levels up
            notebooks_path = os.path.join(current_path, "notebooks")
            if self.debug:
                print(f"DEBUG: Checking for notebooks at: {notebooks_path}")
            
            if os.path.exists(notebooks_path) and os.path.isdir(notebooks_path):
                if self.debug:
                    print(f"DEBUG: Found notebooks directory at: {notebooks_path}")
                return os.path.join(notebooks_path, "out")
            
            # Go up one level
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:  # Reached root
                break
            current_path = parent_path
        
        # Strategy 3: Check if we can find ShearNet directory structure
        current_path = cwd
        for _ in range(5):
            if os.path.exists(os.path.join(current_path, "shearnet")) and \
               os.path.exists(os.path.join(current_path, "shearnet", "core")):
                # We found the ShearNet root directory
                notebooks_path = os.path.join(current_path, "notebooks")
                if self.debug:
                    print(f"DEBUG: Found ShearNet root, using: {notebooks_path}/out")
                return os.path.join(notebooks_path, "out")
            
            parent_path = os.path.dirname(current_path)
            if parent_path == current_path:
                break
            current_path = parent_path
        
        # Fallback: create notebooks/out in current working directory
        fallback_path = os.path.join(cwd, "notebooks", "out")
        if self.debug:
            print(f"DEBUG: Using fallback path: {fallback_path}")
        return fallback_path
    
    def _initialize_output_file(self) -> None:
        """Initialize the markdown output file with header."""
        try:
            if not os.path.exists(self.output_file):
                # File doesn't exist, create it with header
                with open(self.output_file, 'w') as f:
                    f.write(f"# ShearNet Notebook Output\n\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"Output directory: `{self.base_dir}`\n\n")
                    f.write("---\n\n")
                if self.debug:
                    print(f"DEBUG: Created new output file: {self.output_file}")
            else:
                # File exists, just add a session separator
                with open(self.output_file, 'a') as f:
                    f.write(f"\n\n---\n")
                    f.write(f"New session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("---\n\n")
                if self.debug:
                    print(f"DEBUG: Appended to existing output file: {self.output_file}")
        except Exception as e:
            print(f"ERROR: Failed to initialize output file {self.output_file}: {e}")
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message to both console and output file."""
        # Print to console
        print(message)
        
        # Write to file
        try:
            with open(self.output_file, 'a') as f:
                if level == "HEADER":
                    f.write(f"\n## {message}\n\n")
                elif level == "SUBHEADER":
                    f.write(f"\n### {message}\n\n")
                elif level == "CODE":
                    f.write(f"```\n{message}\n```\n\n")
                else:
                    f.write(f"{message}\n\n")
            
            if self.debug and level == "HEADER":
                print(f"DEBUG: Logged to file: {self.output_file}")
        except Exception as e:
            print(f"ERROR: Failed to write to output file: {e}")
    
    def save_plot(self, filename: str, dpi: int = 300, bbox_inches: str = 'tight') -> str:
        """Save the current matplotlib figure and return the path."""
        try:
            # Add timestamp to filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name, ext = os.path.splitext(filename)
            filename = f"{base_name}_{timestamp}{ext}"
            
            filepath = os.path.join(self.base_dir, filename)
            
            if self.debug:
                print(f"DEBUG: Attempting to save plot to: {filepath}")
            
            plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
            
            # Verify the file was created
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"SUCCESS: Plot saved to {filepath} (size: {file_size} bytes)")
                
                # Log the plot save
                self.log(f"![{filename}]({filename})", level="INFO")
                return filepath
            else:
                print(f"ERROR: Plot file was not created: {filepath}")
                return ""
                
        except Exception as e:
            print(f"ERROR: Failed to save plot {filename}: {e}")
            return ""
    
    @contextmanager
    def experiment_section(self, title: str):
        """Context manager for experiment sections."""
        self.log(title, level="HEADER")
        try:
            yield self
        finally:
            self.log("---", level="INFO")

# Global output manager instance
_shearnet_output_manager = None

def get_output_manager() -> ShearNetOutputManager:
    """Get or create the ShearNet output manager."""
    global _shearnet_output_manager
    if _shearnet_output_manager is None:
        _shearnet_output_manager = ShearNetOutputManager(debug=True)
    return _shearnet_output_manager

def log_print(*args, sep: str = ' ', end: str = '\n', level: str = "INFO") -> None:
    """Enhanced print function that logs to file."""
    message = sep.join(str(arg) for arg in args) + end.rstrip()
    get_output_manager().log(message, level)

def save_plot(filename: str, **kwargs) -> str:
    """Save current plot with enhanced error handling."""
    return get_output_manager().save_plot(filename, **kwargs)

def log_array_stats(name: str, array: np.ndarray) -> None:
    """Log statistics about a numpy array."""
    stats = f"{name} stats: shape={array.shape}, min={array.min():.3f}, max={array.max():.3f}, mean={array.mean():.3f}, std={array.std():.3f}"
    log_print(stats, level="CODE")

def experiment_section(title: str):
    """Context manager for experiment sections."""
    return get_output_manager().experiment_section(title)

def reset_output_manager():
    """Reset the global output manager (useful for testing or changing contexts)."""
    global _shearnet_output_manager
    _shearnet_output_manager = None

def test_output_system():
    """Test the output system to make sure it works."""
    print("Testing ShearNet Output System...")
    
    # Test logging
    log_print("This is a test message")
    log_print("This is a header", level="HEADER")
    
    # Test plot saving
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3], [1, 4, 2])
    plt.title("Test Plot")
    save_plot("test_plot.png")
    plt.close()
    
    print("Output system test complete!")