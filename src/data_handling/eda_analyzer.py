from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

from base import BasePipelineComponent  # pylint: disable=import-error
from utils.config_manager import ConfigManager  # pylint: disable=import-error
from visualization.eda_visualizer import EDAVisualizer  # pylint: disable=import-error

ERR_MSG = "EDAnalyzer has not been run yet. First, call the `run()` method or call the instance."



class EDAnalyzer(BasePipelineComponent):
    """A pipeline component that performs exploratory data analysis.

    Computes descriptive statistics, data quality metrics, correlations and IQR outliers.
    Optionally renders plots and saves the results as JSON reports.
    The results are available through read-only properties after `run()`.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Reads the column lists and creates the visualizer if plotting is enabled.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Component name.
        """
        super().__init__(config_manager, name)
        self._basic_stats: Optional[pd.DataFrame] = None
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._quality_report: Optional[Dict[str, Any]] = None
        self._outlier_counts: Optional[pd.Series] = None

        self._target_cols: List[str] = self.config_manager.get_param("data.target_columns", [])
        self._feature_cols: List[str] = self.config_manager.get_param("data.feature_columns", [])
        self._visualizer: Optional[EDAVisualizer] = (
            EDAVisualizer(config_manager, name=f"{self.name}Visualizer")
            if self.config_manager.get_param("pipeline.eda.generate_plots", True)
            else None
        )



    @property
    def basic_stats(self) -> pd.DataFrame:
        """Provides the descriptive statistics table.

        Raises:
            RuntimeError: If `run()` has not been called yet.
        """
        if self._basic_stats is None:
            raise RuntimeError(ERR_MSG)
        return self._basic_stats


    @property
    def correlation_matrix(self) -> pd.DataFrame:
        """Provides the correlation matrix of the non-constant columns.

        Raises:
            RuntimeError: If `run()` has not been called yet.
        """
        if self._correlation_matrix is None:
            raise RuntimeError(ERR_MSG)
        return self._correlation_matrix


    @property
    def quality_report(self) -> Dict[str, Any]:
        """Provides the data quality report.

        Raises:
            RuntimeError: If `run()` has not been called yet.
        """
        if self._quality_report is None:
            raise RuntimeError(ERR_MSG)
        return self._quality_report


    @property
    def outlier_counts(self) -> pd.Series:
        """Provides the number of IQR outliers per numeric non-constant column.

        Raises:
            RuntimeError: If `run()` has not been called yet.
        """
        if self._outlier_counts is None:
            raise RuntimeError(ERR_MSG)
        return self._outlier_counts



    @staticmethod
    def _compute_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Computes descriptive statistics for each numeric column.

        Args:
            df (pd.DataFrame): The analyzed data.

        Returns:
            pd.DataFrame: One row per column with the `describe()` metrics, skewness,
            kurtosis, number of unique values (NaN counts as a value) and missing count.
        """
        stats = df.describe().T
        stats["skew"] = df.skew(numeric_only=True)
        stats["kurtosis"] = df.kurtosis(numeric_only=True)
        stats["unique"] = df.nunique(dropna=False)
        stats["missing_count"] = df.isna().sum()
        return stats


    @staticmethod
    def _check_data_quality(df: pd.DataFrame, stats: pd.DataFrame) -> Dict[str, Any]:
        """Collects data quality metrics.

        Args:
            df (pd.DataFrame): The analyzed data.
            stats (pd.DataFrame): Output of `_compute_basic_stats`, reused for the
                missing and unique counts.

        Returns:
            Dict[str, Any]: Row and column counts, missing values, duplicate rows
            and the names of constant columns.
        """
        missing = stats["missing_count"]
        return {
            "num_rows": len(df),
            "num_columns": df.shape[1],
            "total_missing": int(missing.sum()),
            "missing_by_column": missing[missing > 0].to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "constant_columns": stats.index[stats["unique"] <= 1].to_list(),
        }



    def _analyze_correlations(
        self, df: pd.DataFrame, constant_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Computes the correlation matrix and finds strongly correlated column pairs.

        Args:
            df (pd.DataFrame): The analyzed data.
            constant_cols (List[str]): Columns excluded from the analysis.

        Returns:
            Tuple[pd.DataFrame, List[Dict[str, Any]]]: The correlation matrix and the pairs
            whose absolute correlation reaches `pipeline.eda.correlation_threshold`.
        """
        method = self.config_manager.get_param("pipeline.eda.correlation_method", "pearson")
        threshold = self.config_manager.get_param("pipeline.eda.correlation_threshold", 0.8)

        corr_matrix = df.drop(columns=constant_cols).corr(method=method, numeric_only=True)

        values = corr_matrix.to_numpy()
        rows, cols = np.triu_indices_from(values, k=1)
        pair_values = values[rows, cols]
        strong = np.abs(pair_values) >= threshold

        names: pd.Index = corr_matrix.columns
        high_corr_pairs: list[dict[str, Any]] = [
            {
                "feature_1": str(names[i]),
                "feature_2": str(names[j]),
                "correlation": round(float(value), 6),
            }
            for i, j, value in zip(rows[strong], cols[strong], pair_values[strong])
        ]
        return corr_matrix, high_corr_pairs


    def _detect_outliers_iqr(self, df: pd.DataFrame, constant_cols: List[str]) -> pd.Series:
        """Counts outliers per numeric column using the interquartile range rule.

        A value is an outlier if it lies more than `pipeline.eda.iqr_multiplier` times
        the IQR below the first or above the third quartile.

        Args:
            df (pd.DataFrame): The analyzed data.
            constant_cols (List[str]): Columns excluded from the analysis.

        Returns:
            pd.Series: Number of outliers indexed by column name.
        """
        multiplier = self.config_manager.get_param("pipeline.eda.iqr_multiplier", 1.5)
        numeric = df.drop(columns=constant_cols).select_dtypes(include="number")

        values = numeric.to_numpy()
        q1, q3 = np.nanquantile(values, [0.25, 0.75], axis=0)
        margin = multiplier * (q3 - q1)
        is_outlier = (values < q1 - margin) | (values > q3 + margin)
        return pd.Series(is_outlier.sum(axis=0), index=numeric.columns)


    def _plot(self, df: pd.DataFrame, constant_cols: List[str]) -> None:
        """Renders distributions, correlation matrix and boxplots, if plotting is enabled.

        Args:
            df (pd.DataFrame): The analyzed data.
            constant_cols (List[str]): Columns excluded from the plots.
        """
        if self._visualizer is None:
            return

        constant = set(constant_cols)
        targets = df[[c for c in self._target_cols if c not in constant]]
        features = df[[c for c in self._feature_cols if c not in constant]]

        self._visualizer.plot_distributions(targets, "target_columns")
        self._visualizer.plot_distributions(features, "feature_columns")
        self._visualizer.plot_correlation_matrix(self.correlation_matrix)
        self._visualizer.plot_boxplots(targets, "target_columns")
        self._visualizer.plot_boxplots(features, "feature_columns")


    def _save_reports(self) -> None:
        """Saves the quality report, statistics and correlations as JSON files.

        Does nothing if `pipeline.eda.save_reports` is false. The files are written
        to the `eda` subdirectory of `settings.paths.output_dir`.
        """
        if not self.config_manager.get_param("pipeline.eda.save_reports", True):
            return

        eda_dir = Path(self.config_manager.get_param("settings.paths.output_dir", "output")) / "eda"
        eda_dir.mkdir(parents=True, exist_ok=True)

        reports: Dict[str, Any] = {
            "dataset_info": self.quality_report,
            "basic_statistics": json.loads(self.basic_stats.to_json(orient="index")),
            "correlation_matrix": json.loads(self.correlation_matrix.to_json(orient="index")),
        }
        for key, report in reports.items():
            with open(eda_dir / f"{key}.json", mode="w", encoding="utf-8") as file:
                json.dump(report, file, indent=4, ensure_ascii=False)



    def run(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """Runs the exploratory analysis, renders the plots and saves the reports.

        Args:
            df (pd.DataFrame): The data to analyze.

        Returns:
            Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]: The quality report,
            the descriptive statistics and the correlation matrix.
        """
        self._basic_stats = self._compute_basic_stats(df)
        quality_report = self._check_data_quality(df, self._basic_stats)
        constant_cols = quality_report["constant_columns"]

        self._correlation_matrix, quality_report["high_correlations"] = (
            self._analyze_correlations(df, constant_cols)
        )
        self._outlier_counts = self._detect_outliers_iqr(df, constant_cols)
        quality_report["outliers_iqr"] = self._outlier_counts.to_dict()
        self._quality_report = quality_report

        self._plot(df, constant_cols)
        self._save_reports()
        return self._quality_report, self._basic_stats, self._correlation_matrix
