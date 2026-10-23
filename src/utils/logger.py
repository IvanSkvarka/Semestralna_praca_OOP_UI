from typing import Union
from pathlib import Path
import logging
from .config_manager import ConfigManager

def get_logger(name: str, config_manager: Union[ConfigManager, None] = None) -> logging.Logger:
    """
    
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    if config_manager:
        level_str = config_manager.get_param("settings.logging.level", default="INFO")
        log_format = config_manager.get_param(
            "settings.logging.format",default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        log_dir = config_manager.get_param("settings.paths.log_dir", default=None)
    else:
        level_str = "INFO"
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        log_dir = "reports/logs"
    # Zmena zo stringu (musi byt uppercase) na numericke ohodnotenie
    level_num = getattr(logging, level_str.upper(), logging.INFO)

    logger.setLevel(level_num)
    logger.propagate = False # Zabranenie zdvojnasobovaniu sprav
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"{name}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    else:
        import sys # pylint: disable=import-outside-toplevel
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
