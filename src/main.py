"""Entry point of the ML pipeline. Run it from the repository root: python src/main.py"""
from typing import List
from utils.config_manager import ConfigManager
from data_handling.data_loader import DataLoader
from data_handling.eda_analyzer import EDAnalyzer
from data_handling.preprocessor import Preprocessor
from models import meanbaseline_model, logloglinear_model, gradientboosting_model, base_model
from evaluation.evaluator import Evaluator
from evaluation.tuner import HyperparameterTuner


if __name__ == "__main__":
    cfg = ConfigManager("config")
    cfg.load_config(["settings", "data", "pipeline", "visualization", "dictionary"])

    data_loader = DataLoader(cfg, "ReactorDataLoader")
    df = data_loader()

    eda_analyzer = EDAnalyzer(cfg, "ReactorEDAnalyzer")
    eda_analyzer(df)

    preprocessor = Preprocessor(cfg, "ReactorPreprocessor")
    X, y, groups = preprocessor(df)

    eval_models: List[base_model.BaseModel] = [
        meanbaseline_model.MeanBaselineModel(cfg),
        logloglinear_model.LogLogLinearModel(cfg),
        gradientboosting_model.GradientBoostingModel(cfg)
    ]

    tuner = HyperparameterTuner(cfg, "ReactorTuner")
    eval_models = tuner(eval_models)

    evaluator = Evaluator(cfg, "ReactorEvaluator")
    evaluator(eval_models, X, y, groups)
