# Semestrálna práca

Návrh a implementácia znovupoužiteľného objektovo orientovaného pipeline v Pythone, ktorý načíta surové dáta, overí ich kvalitu, upraví ich a natrénuje regresné modely na predikciu veličiny.

## Téma práce

Navrhnutý pipeline je riadený konfiguračnými súbormi YAML, takže vstupné stĺpce, predspracovanie, modely a mriežku hyperparametrov možno meniť bez zásahu do kódu.

Riešená úloha je regresia. Dátová sada obsahuje 768 záznamov a 24 stĺpcov: 20 vstupných príznakov (geometria miešaného reaktora, parametre miešadiel a otáčky rpm) a 4 cieľové veličiny. Predikovanou veličinou je medián turbulentnej disipácie epsilon_p50, ktorý sa pohybuje v rozsahu od 4 * 10^(-4) do 0,73. [https://www.kaggle.com/datasets/novalabs/stirred-reactor-cfd-ml-dataset]

Porovnávajú sa tri modely:

- MeanBaselineModel: referenčný model, ktorý predikuje priemer cieľovej veličiny.
- LogLogLinearModel: lineárna regresia v log-log tvare (mocninový zákon).
- GradientBoostingModel: gradient boosting s ladením hyperparametrov.

## Štruktúra projektu

### Strom priečinkov
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
│   ├── api
│   │   ├── core.md
│   │   ├── data_handling.md
│   │   ├── evaluation.md
│   │   ├── models.md
│   │   ├── tests.md
│   │   ├── utils.md
│   │   └── visualization.md
│   └── index.md
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
├── tests
│   ├── configuration.py
│   ├── run_all_tests.py
│   ├── test_config.py
│   ├── test_data_handling.py
│   ├── test_evaluation.py
│   └── test_models.py
├── LICENSE
├── mkdocs.yml
├── README.md
└── requirements.txt
```

### Popis priečinkov

`src/`:
`config/`:
`data/`:
`tests/`:
`report/`:
`docs/`:

## Inštalácia a spustenie

`python -m venv ./.venv`

`pip install -r requirements.txt`

[!WARNING]
**Special Requirements:** Pred inštaláciou sa uistite, že máte nainštalovaný `LaTeX` a `cm-super`. V opačnom prípade nastavte `use_latex` = `False`.

`python src/main.py`



## Konfigurácia

## Architektúra

## Metodika

1. Analýza dát (EDA). EDAnalyzer vypočíta popisné štatistiky, počet chýbajúcich hodnôt a duplicít, konštantné stĺpce, korelácie (Pearson, prah 0,85) a odľahlé hodnoty metódou IQR (násobok 1,5 kvartilov). Odhalí napríklad konštantné stĺpce alebo duplicitné riadky.

2. Predspracovanie. Preprocessor odstráni duplicity a konštantné stĺpce. Vybraný cieľ a príznaky sa logaritmujú (log10), pretože pre veličiny meniace sa v rádoch a fyzikálne vzťahy majú mocninový charakter. Model teda pracuje s log10 hodnotou cieľa.

3. Skupiny a delenie dát. Záznamy patria do geometrických rodín, určených kombináciou parametrov. Riadky jednej rodiny sú si veľmi podobné, takže pri náhodnom delení by sa rodina dostala do trénovacej aj testovacej časti a skóre by bolo nadhodnotené. Preto sa používa GroupKFold. Rodina je vždy celá buď v trénovacej, alebo v testovacej časti.

4. Vnorená krížová validácia.

Vonkajšia slučka: GroupKFold s `n` zloženiami meria výkon na dátach, ktoré model nevidel.
Vnútorná slučka: GridSearchCV s GroupKFold (`m` zložeí) volí hyperparametre iba z trénovacej časti vonkajšieho zloženia, takže testovacie dáta neovplyvnia výber.
Hľadá sa najnižšie RMSE (`neg_root_mean_squared_error`) a po výbere sa model znova natrénuje na celej trénovacej časti.

5. Hyperparametre Gradient Boostingu. Mriežka cez všetky zadané kombinácie: `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, `subsample`. Ostatné modely sa neladia.

6. Metriky. Na log10 škále sa počíta R2, RMSE a MAE, na pôvodnej škále RMSE a MAE (po spätnej transformácii 10^x). Hodnotia sa predikcie zo všetkých zložení dokopy, priemer a smerodajná odchýlka cez zloženia. Modely sa radia podľa R2 na log10 škále.

## Testy

## Autor a licencia
Autor: Ivan Škvarka, Strojní fakulta ČVUT v Prahe, odbor Automatizace a průmyslová informatika (magisterské štúdium, 1. ročník).
Predmet: Umělá inteligence a Objektově orientované programování, akademický rok 2026/2027.

Licencia: Projekt je vydaný pod licenciou MIT, pozri súbor LICENSE. Copyright © 2026 IvanSkvarka.