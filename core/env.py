from pathlib import Path
import os

DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def load_env_file(path=None):
    env_path = Path(path) if path else DEFAULT_ENV_FILE
    if not env_path.exists():
        return env_path

    try:
        content = env_path.read_text(encoding="utf-8")
    except Exception:
        content = env_path.read_text()

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith("\"") and value.endswith("\"")) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value

    return env_path


def get_env(key, default=None, required=False):
    value = os.getenv(key)
    if value is None or value == "":
        if required:
            raise RuntimeError(f"Missing required env var: {key}")
        return default
    return value


def get_env_int(key, default=None, required=False):
    value = get_env(key, default=None, required=required)
    if value is None or value == "":
        return default
    return int(value)


def get_env_float(key, default=None, required=False):
    value = get_env(key, default=None, required=required)
    if value is None or value == "":
        return default
    return float(value)


def get_env_bool(key, default=None, required=False):
    value = get_env(key, default=None, required=required)
    if value is None or value == "":
        return default
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    if default is None:
        raise RuntimeError(f"Invalid boolean env var: {key}={value!r}")
    return default


def get_env_list(key, default=None, required=False, delimiter=","):
    value = get_env(key, default=None, required=required)
    if value is None or value == "":
        return default or []
    return [item.strip() for item in value.split(delimiter) if item.strip()]


def get_env_path(key, base_dir=None, default=None, required=False):
    value = get_env(key, default=None, required=required)
    if value is None or value == "":
        return default
    path = Path(value)
    if base_dir and not path.is_absolute():
        return Path(base_dir) / value
    return path
