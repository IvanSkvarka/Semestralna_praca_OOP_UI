""""
Main module
"""

from data_handling.data_loader import DataLoader
from data_handling.eda_analyzer import EDAnalyzer
from utils.config_manager import ConfigManager
from visualization.eda_visualizer import EDAVisualizer



if __name__ == "__main__":
    cfg = ConfigManager("config")
    cfg.load_config(["settings", "data", "pipeline", "visualization", "dictionary"])

    data_loader = DataLoader(cfg, "ReactorDataLoader")
    data_loader()
    
    eda_analyzer = EDAnalyzer(cfg, "ReactorEDAnalyzer")
    eda_analyzer(data_loader.data)

    eda_visualizer = EDAVisualizer(cfg, "ReactorEDAVisualizer")
    eda_visualizer(
        data_loader.data,
        eda_analyzer.correlation_matrix,
    )