# Semestrálna práca

## Téma práce
Návrh a implementácia znovupoužiteľného objektovo orientovaného systému (pipeline), ktorý načíta surové dáta,
vykoná ich validáciu, transformáciu a natrénuje regresný model schopný predikovať veličiny.

## Štruktúra projektu
<pre>
.
├── config
│   ├── config.yaml
│   └── settings.yaml
├── data
│   └── training_data_set.csv
├── docs
├── report
│   └── logs
├── src
│   ├── data_handling
│   │   ├── data_loader.py
│   │   └── __init__.py
│   ├── utils
│   │   ├── base.py
│   │   ├── config_manager.py
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── base.py
│   ├── __init__.py
│   └── main.py
├── tests
├── LICENSE
└── README.md
</pre>