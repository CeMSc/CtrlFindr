"""CtrlFindr: dictionary-based content analysis of text documents, sentence by sentence."""

__version__ = "3.0.0"

from .config import Config, config_from_dict, load_config  # noqa: E402

__all__ = ["Config", "config_from_dict", "load_config", "__version__"]
