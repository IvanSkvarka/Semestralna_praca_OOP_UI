"""Entry point of the ML pipeline. Run it from the repository root: python src/main.py"""

from data_handling.data_loader import DataLoader
from data_handling.eda_analyzer import EDAnalyzer
from utils.config_manager import ConfigManager



if __name__ == "__main__":
    cfg = ConfigManager("config")
    cfg.load_config(["settings", "data", "pipeline", "visualization", "dictionary"])

    data_loader = DataLoader(cfg, "ReactorDataLoader")
    df = data_loader()

    eda_analyzer = EDAnalyzer(cfg, "ReactorEDAnalyzer")
    eda_analyzer(df)
