from pathlib import Path
import pandas as pd
from base import BasePipelineComponent
from config import ConfigManager
from typing import Union

class ReactorDataLoader(BasePipelineComponent):
    """
    
    """

    def __init__(self, config_manager: ConfigManager, name: Union[str, None] = None) -> None:
        """
        
        """
        super().__init__(config_manager, name)
        self._file_path = Path(self.config_manager["paths.raw_data"])
        self._target_columns = list(self.config_manager["data.target_columns"])
        # self._feature_columns = list(self.config_manager["data.feature_columns"])

        self._data: pd.DataFrame = None

    # # 3. INTERNÁ POMOCNÁ METÓDA NA VALIDÁCIU SCHÉMY
    # METÓDA _validate_dataframe(self, df: pd.DataFrame) -> None:
    #     """
    #     Overí, že načítaný DataFrame nie je prázdny a obsahuje všetky požadované stĺpce.
    #     """
    #     AK JE df prázdny (df.empty):
    #         VYHOĎ ValueError("Načítaný dataset je prázdny.")

    #     # Zisti, ktoré požadované cieľové stĺpce v df chýbajú
    #     chybajuce_ciele = [col PRE KAŽDÝ col V self._target_columns AK col NIE JE V df.columns]
    #     AK chybajuce_ciele NIE JE prázdny zoznam:
    #         VYHOĎ ValueError(f"V datasete chýbajú cieľové stĺpce: {chybajuce_ciele}")

    #     # Ak boli v configu zadané aj explicitné feature_columns, skontroluj aj tie
    #     AK self._feature_columns NIE JE None:
    #         chybajuce_priznaky = [col PRE KAŽDÝ col V self._feature_columns AK col NIE JE V df.columns]
    #         AK chybajuce_priznaky NIE JE prázdny zoznam:
    #             VYHOĎ ValueError(f"V datasete chýbajú vstupné stĺpce: {chybajuce_priznaky}")

    def run(self) -> pd.DataFrame:
        """
        
        """

        df = pd.read_csv(self._file_path)
        print(df)

        self._data = df
        return self._data