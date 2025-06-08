# shearnet/config/config_handler.py
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import copy

class Config:
    """Handle configuration loading and command-line overrides."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config handler."""
        self.default_config_path = Path(__file__).parent / "default_config.yaml"
        self.config = self._load_config(config_path)
        self._setup_paths()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        # Load default config first
        with open(self.default_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # If custom config provided, update defaults
        if config_path is not None:
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
            # Deep merge custom config into default
            self._deep_merge(config, custom_config)
        
        return config
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Deep merge update dict into base dict."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _setup_paths(self) -> None:
        """Setup default paths based on environment variables."""
        data_path = os.getenv('SHEARNET_DATA_PATH', os.path.abspath('.'))
        
        if self.config['output']['save_path'] is None:
            self.config['output']['save_path'] = os.path.join(data_path, 'model_checkpoint')
        
        if self.config['output']['plot_path'] is None:
            self.config['output']['plot_path'] = os.path.join(data_path, 'plots')
        
        # Ensure paths exist
        os.makedirs(self.config['output']['save_path'], exist_ok=True)
        os.makedirs(self.config['output']['plot_path'], exist_ok=True)
    
    def update_from_args(self, args: argparse.Namespace) -> None:
        """Update config with command-line arguments."""
        args_dict = vars(args)
        
        # Get the mapping for training mode
        mapping = self._get_train_mapping()
        
        # Update config with non-None arguments
        for arg_name, config_path in mapping.items():
            if arg_name in args_dict and args_dict[arg_name] is not None:
                self._set_nested(config_path, args_dict[arg_name])
    
    def _get_train_mapping(self) -> Dict[str, str]:
        """Get argument mapping for training mode."""
        return {
            # Dataset args
            'samples': 'dataset.samples',
            'psf_sigma': 'dataset.psf_sigma',
            'exp': 'dataset.exp',
            'seed': 'dataset.seed',
            
            # Model args
            'nn': 'model.type',
            
            # Training args
            'epochs': 'training.epochs',
            'batch_size': 'training.batch_size',
            'learning_rate': 'training.learning_rate',
            'weight_decay': 'training.weight_decay',
            'patience': 'training.patience',
            
            # Output args
            'save_path': 'output.save_path',
            'plot_path': 'output.plot_path',
            'model_name': 'output.model_name',
            
            # Plotting args
            'plot': 'plotting.plot',
        }
    
    def _set_nested(self, path: str, value: Any) -> None:
        """Set nested config value using dot notation."""
        keys = path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get config value using dot notation."""
        keys = path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except KeyError:
            return default
    
    def save(self, path: str) -> None:
        """Save current config to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def print_config(self) -> None:
        """Print current configuration."""
        print("\n" + "="*50)
        print("Training Configuration")
        print("="*50)
        
        for section in ['dataset', 'model', 'training', 'output', 'plotting']:
            if section in self.config:
                print(f"\n{section}:")
                for key, value in self.config[section].items():
                    print(f"  {key}: {value}")
        print("="*50 + "\n")

    def print_eval_config(self) -> None:
        """Print only evaluation-relevant configuration."""
        print("\n" + "="*50)
        print("Evaluation Configuration")
        print("="*50)
        
        # Only print relevant sections for evaluation
        sections_to_print = ['evaluation', 'model', 'plotting', 'comparison']
        
        for section in sections_to_print:
            if section in self.config:
                print(f"\n{section}:")
                for key, value in self.config[section].items():
                    print(f"  {key}: {value}")
        print("="*50 + "\n")