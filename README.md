# Semestrálna práca

Návrh a implementácia znovupoužiteľného objektovo orientovaného pipeline v Pythone, ktorý načíta surové dáta, overí ich kvalitu, upraví ich a natrénuje regresné modely na predikciu veličiny.

## Téma práce

Navrhnutý pipeline je riadený konfiguračnými súbormi YAML, takže vstupné stĺpce, predspracovanie, modely a mriežku hyperparametrov možno meniť bez zásahu do kódu.

Riešená úloha je regresia. Dátová sada obsahuje 768 záznamov a 24 stĺpcov: 20 vstupných príznakov (geometria miešaného reaktora, parametre miešadiel a otáčky rpm) a 4 cieľové veličiny. Predikovanou veličinou je medián turbulentnej disipácie epsilon_p50, ktorý sa pohybuje v rozsahu od 4 * 10^(-4) do 0,73 [Stirred reactor dataset](https://www.kaggle.com/datasets/novalabs/stirred-reactor-cfd-ml-dataset).

Porovnávajú sa tri modely:

- MeanBaselineModel: referenčný model, ktorý predikuje priemer cieľovej veličiny.
- LogLogLinearModel: lineárna regresia v log-log tvare (mocninový zákon).
- GradientBoostingModel: gradient boosting s ladením hyperparametrov.

## Štruktúra projektu

### Strom priečinkov
    ```bash
    .
    ├── config
    ├── data
    ├── docs
    │   └── api
    ├── report
    │   ├── eda
    │   ├── evaluation
    │   ├── figures
    │   │   ├── eda
    │   │   ├── evaluation
    │   │   └── preprocessing
    │   └── preprocessing
    ├── src
    │   ├── data_handling
    │   ├── evaluation
    │   ├── models
    │   ├── utils
    │   └── visualisation
    ├── tests
    ├── LICENSE
    ├── mkdocs.yml
    ├── README.md
    └── requirements.txt
    ```

### Popis priečinkov

`src/`: Adresár so zdrojovými kódmi.
`config/`: Konfiguračný adresár. Viac v časti **Konfigurácia**.
`data/`: Adresár obsahujúci dáta.
`tests/`: Testovacie skripty pre overenie funkčnosti modulov v `src/`. Viac v časti **Testy**.
`report/`: Priečinok s výsledkoami vo formáte JSON a grafmi.
`docs/`: Dokumentácia projektu.

## Inštalácia a spustenie

Naklonujete repozitár:
    ```bash
    git clone https://github.com/IvanSkvarka/Semestralna_praca_OOP_UI.git
    cd Semestralna_praca_OOP_UI
    ```

> [!TIP]
> Pre nezavislý projekt vytvorte virtuálne prostredie (virtual enviroment):
>   ```bash
>   python -m venv ./.venv
>   source .venv/bin/activate # pre Windows: venv\Scripts\activate
>   ```

Nainštalujte závislosti:
    ```bash
    pip install -r requirements.txt
    ```

> [!WARNING]
> **Special Requirements:** Pred inštaláciou sa uistite, že máte nainštalovaný `LaTeX` a `cm-super`. V opačnom prípade nastavte `use_latex` = `False`.

> [!TIP]
> **Konfigurácia projektu:** Projekt je spustiteľný s prednastavenou konfiguráciou, avšak pre personalizované výsledky, prípadne výsledky pre iný dataset treba konfiguráciu nastaviť.


Pri spustení sa musíte nachádzať v koreňovom adresáry, potom použite príkaz:
    ```bash
    python src/main.py
    ```

Dokumentácia je dostupná po použití príkazu z koreňového adresára:
    ```bash
    mkdocs serve
    ```

## Konfigurácia

## Princípy OOP

### Abstrakcia
### Dedičnosť
    ```bash
    BaseConfigManager
    └── ConfigManager

    BasePipelineComponent
    ├── DataLoader
    ├── EDAnalyzer
    ├── Preprocessor
    ├── HyperparameterTuner
    ├── Evaluator
    ├── BaseVisualizer
    │   ├── EDAVisualizer
    │   ├── PreprocessorVisualizer
    │   └── EvaluationVisualizer
    └── BaseModel   (spolu so sklearn BaseEstimator, RegressorMixin)
        ├── MeanBaselineModel
        ├── LogLogLinearModel
        └── GradientBoostingModel
    ```
### Polymorfizmus
### Zapuzdrenie
### Iné:
Dependecy Injection, Template Method, Pretazenie operatorov, vzor Strategy, Prepisanie metody.

### Tok dát
DataLoader -> EDAnalyzer -> Preprocessor -> HyperparameterTuner -> Evaluator.

## Metodika

1. Analýza dát (EDA). EDAnalyzer vypočíta popisné štatistiky, počet chýbajúcich hodnôt a duplicít, konštantné stĺpce, korelácie (Pearson, prah 0,85) a odľahlé hodnoty metódou IQR (násobok 1,5 kvartilov). Odhalí napríklad konštantné stĺpce alebo duplicitné riadky.

2. Predspracovanie. Preprocessor odstráni duplicity a konštantné stĺpce. Vybraný cieľ a príznaky sa logaritmujú (log10), pretože pre veličiny meniace sa v rádoch a fyzikálne vzťahy majú mocninový charakter. Model teda pracuje s log10 hodnotou cieľa.

3. Skupiny a delenie dát. Záznamy patria do geometrických rodín, určených kombináciou parametrov. Riadky jednej rodiny sú si veľmi podobné, takže pri náhodnom delení by sa rodina dostala do trénovacej aj testovacej časti a skóre by bolo nadhodnotené. Preto sa používa GroupKFold. Rodina je vždy celá buď v trénovacej, alebo v testovacej časti.

4. Vnorená krížová validácia.

- Vonkajšia slučka: GroupKFold s `n` zloženiami meria výkon na dátach, ktoré model nevidel.
- Vnútorná slučka: GridSearchCV s GroupKFold (`m` zložení) volí hyperparametre iba z trénovacej časti vonkajšieho zloženia, takže testovacie dáta neovplyvnia výber.
- Hľadá sa najnižšie RMSE (`neg_root_mean_squared_error`) a po výbere sa model znova natrénuje na celej trénovacej časti.

5. Hyperparametre Gradient Boostingu. Mriežka cez všetky zadané kombinácie: `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, `subsample`. Ostatné modely sa neladia.

6. Metriky. Na log10 škále sa počíta R2, RMSE a MAE, na pôvodnej škále RMSE a MAE (po spätnej transformácii 10^x). Hodnotia sa predikcie zo všetkých zložení dokopy, priemer a smerodajná odchýlka cez zloženia. Modely sa radia podľa R2 na log10 škále.

## Testy
Spustenie testov:
    ```bash
    python tests/run_all_tests.py
    ```
## Autor a licencia
Autor: Ivan Škvarka, Strojní fakulta ČVUT v Praze, odbor Automatizace a průmyslová informatika (magisterské štúdium, 1. ročník).
Predmet: Umělá inteligence a Objektově orientované programování, akademický rok 2026/2027.

Licencia: Projekt je vydaný pod licenciou MIT, pozri súbor LICENSE. Copyright © 2026 IvanSkvarka.