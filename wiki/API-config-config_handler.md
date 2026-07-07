# `shearnet.config.config_handler`

> Module: `shearnet.config.config_handler`

Layered YAML + command-line configuration handling for ShearNet.

<a id="shearnet.config.config_handler.Config"></a>

## `Config`

```python
class Config()
```

Layered configuration for the ShearNet CLIs.

Loads ``config/default_config.yaml`` first, then deep-merges an optional
user YAML on top, and finally lets command-line arguments override
individual values. Output paths default to ``$SHEARNET_DATA_PATH`` (or the
current directory) and are created on init.

Values are read and written with dot-notation paths, e.g.
``config.get('training.epochs')`` or ``config._set_nested('dataset.seed', 0)``.

<a id="shearnet.config.config_handler.Config.__init__"></a>

#### `__init__`

```python
def __init__(config_path: Optional[str] = None)
```

Initialize the config, optionally merging the YAML at ``config_path``.

<a id="shearnet.config.config_handler.Config.update_from_args"></a>

#### `update_from_args`

```python
def update_from_args(args: argparse.Namespace) -> None
```

Update config with command-line arguments.

<a id="shearnet.config.config_handler.Config.get"></a>

#### `get`

```python
def get(path: str, default: Any = None) -> Any
```

Get config value using dot notation.

<a id="shearnet.config.config_handler.Config.save"></a>

#### `save`

```python
def save(path: str) -> None
```

Save current config to file.

<a id="shearnet.config.config_handler.Config.print_config"></a>

#### `print_config`

```python
def print_config() -> None
```

Print current configuration.

<a id="shearnet.config.config_handler.Config.print_eval_config"></a>

#### `print_eval_config`

```python
def print_eval_config() -> None
```

Print only evaluation-relevant configuration.
