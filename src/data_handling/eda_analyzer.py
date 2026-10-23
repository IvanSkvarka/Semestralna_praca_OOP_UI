from typing import Union, Optional, Any, Dict, Tuple, List
from pathlib import Path
import json
import pandas as pd

from base import BasePipelineComponent # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error



ERR_MSG = "EDAnalyzer has not been run yet. " \
          "First, call the `run()` method or an instance of the object."



class EDAnalyzer(BasePipelineComponent):
    """
    
    """
    def __init__(self, config_manager: ConfigManager, name: Union[str, None] = None) -> None:
        """
        
        """
        super().__init__(config_manager, name)
        self._basic_stats: Optional[pd.DataFrame] = None
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._quality_report: Optional[Dict[str, Any]] = None
        self._outlier_counts: Optional[pd.Series] = None



    @property
    def basic_stats(self) -> pd.DataFrame:
        """ """
        if self._basic_stats is None:
            self.logger.error(ERR_MSG)
            raise RuntimeError(ERR_MSG)
        return self._basic_stats


    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """ """
        if self._correlation_matrix is None:
            self.logger.error(ERR_MSG)
            raise RuntimeError(ERR_MSG)
        return self._correlation_matrix

    @property
    def quality_report(self) -> Dict[str, Any]:
        """ """
        if self._quality_report is None:
            self.logger.error(ERR_MSG)
            raise RuntimeError(ERR_MSG)
        return self._quality_report


    @property
    def outlier_counts(self) -> pd.Series:
        """ """
        if self._outlier_counts is None:
            self.logger.error(ERR_MSG)
            raise RuntimeError(ERR_MSG)
        return self._outlier_counts



    def _compute_basic_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        
        """
        stats = df.describe().T
        stats["skew"] = df.skew(numeric_only=True)
        stats["kurtosis"] = df.kurtosis(numeric_only=True)
        stats["unique"] = df.nunique()
        stats["missing_count"] = df.isna().sum()

        return stats


    def _check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        
        """
        total_missing = int(df.isna().sum().sum())
        missing_by_col = df.isna().sum()[df.isna().sum() > 0].to_dict()
        duplicate_rows = int(df.duplicated().sum())
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

        return {
            "num_rows": len(df),
            "num_columns": df.shape[1],
            "total_missing": total_missing,
            "missing_by_column": missing_by_col,
            "duplicate_rows": duplicate_rows,
            "constant_columns": constant_cols,
        }


    def _analyze_correlations(
            self, df: pd.DataFrame, constant_cols: List[str]
        ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        
        """
        method = self.config_manager.get_param("config.eda.correlation_method", "pearson")
        threshold = self.config_manager.get_param("config.eda.correlation_threshold", 0.95)

        df_valid = df.drop(columns=constant_cols)
        corr_matrix = df_valid.corr(method=method, numeric_only=True)
        high_corr_pairs: List[Dict[str, Any]] = []

        for i in range(0, corr_matrix.shape[1]):
            for j in range(i + 1, corr_matrix.shape[1]):
                val = corr_matrix.iloc[i, j]
                if pd.notna(val) and abs(float(val)) >= threshold: # type: ignore
                    high_corr_pairs.append(
                        {
                            "feature_1": str(corr_matrix.columns[i]),
                            "feature_2": str(corr_matrix.columns[j]),
                            "correlation": round(float(val), 6), # type: ignore
                        }
                    )

        return (corr_matrix, high_corr_pairs)


    def _detect_outliers_iqr(self, df: pd.DataFrame, constant_cols: List[str]) -> pd.Series:
        """
        
        """
        multiplier = self.config_manager.get_param("config.eda.iqr_multiplier", 1.5)

        df_valid = df.drop(columns=constant_cols)
        df_numeric = df_valid.select_dtypes(include="number")

        q1 = df_valid.quantile(0.25)
        q3 = df_valid.quantile(0.75)
        iqr = q3 - q1

        is_outlier = (df_numeric < (q1 - multiplier * iqr)) | (
            df_numeric > (q3 + multiplier * iqr)
        )
        return is_outlier.sum()


    def _save_reports(self) -> None:
        """
        
        """
        if not self.config_manager.get_param("config.eda.save_reports", True):
            self.logger.info(
                "Saving is turned off. If you want save reports,"
                " add config.eda.save_reports and set value `True`."
            )
            return None

        base_output = self.config_manager.get_param("settings.paths.output_dir", "report")

        eda_dir = Path(base_output) / "eda"
        eda_dir.mkdir(parents=True, exist_ok=True)

        full_report: Dict[str, Any] = {
            "dataset_info": self._quality_report,
            "basic_statistics": (
                json.loads(self._basic_stats.to_json(orient="index"))
                if self._basic_stats is not None
                else {}
            ),
            "correlation_matrix": (
                json.loads(self._correlation_matrix.to_json(orient="index"))
                if self._correlation_matrix is not None
                else {}
            ),
        }

        for key, item in full_report.items():
            output_path = eda_dir / f"{key}.json"
            with open(output_path, mode="w", encoding="utf-8") as file:
                json.dump(item, file, indent=4, ensure_ascii=False)
        self.logger.info("Output saved in: %s", eda_dir)

        return None

    def run(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """
        
        """
        self._basic_stats = self._compute_basic_stats(df)
        self._quality_report = self._check_data_quality(df)
        (
            self._correlation_matrix, high_correlations
        ) = self._analyze_correlations(
            df, self._quality_report["constant_columns"]
        )
        self._quality_report["high_correlations"] = high_correlations
        self._outlier_counts = self._detect_outliers_iqr(
            df, self._quality_report["constant_columns"]
        )
        self._quality_report["outliers_iqr"] = self._outlier_counts.to_dict()
        self._save_reports()
        return (self.quality_report, self.basic_stats, self.correlation_matrix)
