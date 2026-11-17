# Semestrálna práca

## Téma práce
Návrh a implementácia znovupoužiteľného objektovo orientovaného systému (pipeline), ktorý načíta surové dáta,
vykoná ich validáciu, transformáciu a natrénuje regresný model schopný predikovať veličiny.

## Štruktúra projektu
```bash
.
├── config
│   ├── data.yaml
│   ├── dictionary.yaml
│   ├── pipeline.yaml
│   ├── settings.yaml
│   └── visualization.yaml
├── data
│   └── training_data_set.csv
├── docs
├── report
│   ├── eda
│   │   ├── basic_statistics.json
│   │   ├── correlation_matrix.json
│   │   └── dataset_info.json
│   ├── evaluation
│   │   ├── evaluation_summary.json
│   │   └── GridSearchCV_best_params.json
│   ├── figures
│   │   ├── eda
│   │   │   ├── correlation_matrix.pdf
│   │   │   ├── feature_columns_boxplot.pdf
│   │   │   ├── feature_columns_histplot.pdf
│   │   │   ├── target_columns_boxplot.pdf
│   │   │   └── target_columns_histplot.pdf
│   │   ├── evaluation
│   │   │   ├── metrics_comparison.pdf
│   │   │   └── predicted_vs_true.pdf
│   │   └── preprocessing
│   │       ├── features_histplot.pdf
│   │       ├── group_sizes.pdf
│   │       └── target_histplot.pdf
│   └── preprocessing
│       └── preprocessing_report.json
├── src
│   ├── data_handling
│   │   ├── data_loader.py
│   │   ├── eda_analyzer.py
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   ├── evaluation
│   │   ├── evaluator.py
│   │   ├── __init__.py
│   │   └── tuner.py
│   ├── models
│   │   ├── base_model.py
│   │   ├── gradientboosting_model.py
│   │   ├── __init__.py
│   │   ├── logloglinear_model.py
│   │   └── meanbaseline_model.py
│   ├── utils
│   │   ├── base_config_manager.py
│   │   ├── config_manager.py
│   │   └── __init__.py
│   ├── visualization
│   │   ├── base_visualizer.py
│   │   ├── eda_visualizer.py
│   │   ├── evaluation_visualizer.py
│   │   ├── __init__.py
│   │   └── preprocessor_visualizer.py
│   ├── base.py
│   └── main.py
├── LICENSE
└── README.md
```