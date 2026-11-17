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
│   ├── figures
│   │   ├── eda
│   │   └── preprocessing
│   └── preprocessing
├── src
│   ├── data_handling
│   │   ├── data_loader.py
│   │   ├── eda_analyzer.py
│   │   ├── __init__.py
│   │   └── preprocessor.py
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
│   │   ├── __init__.py
│   │   └── preprocessor_visualizer.py
│   ├── base.py
│   └── main.py
├── LICENSE
└── README.md
```