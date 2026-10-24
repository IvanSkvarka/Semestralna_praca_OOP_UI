""""
Main module
"""

from data_handling.data_loader import DataLoader # pylint: disable=import-error
from data_handling.eda_analyzer import EDAnalyzer # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error



if __name__ == "__main__":
    cfg = ConfigManager("config")
    cfg.load_config(["settings", "config", "units"])
    data_loader = DataLoader(cfg, "ReactorDataLoader")
    data_loader()
    eda_analyzer = EDAnalyzer(cfg, "ReactorEDAnalyzers")
    eda_analyzer(data_loader.data)
