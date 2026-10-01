"""ShearNet configuration: one schema, validated on load."""

from .config_handler import Config, ConfigError, load_config

__all__ = ["Config", "ConfigError", "load_config"]
