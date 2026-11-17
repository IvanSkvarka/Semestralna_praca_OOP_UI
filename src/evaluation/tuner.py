from typing import Any, Dict, List, Optional, Sequence, Union

from sklearn.model_selection import GridSearchCV, GroupKFold

from base import BasePipelineComponent # pylint: disable=import-error
from utils.config_manager import ConfigManager # pylint: disable=import-error
from models.base_model import BaseModel # pylint: disable=import-error


class HyperparameterTuner(BasePipelineComponent):
    """A pipeline component that prepares the hyperparameter search of the models.

    Wraps each model that has a parameter grid in `pipeline.tuning.param_grids` into a
    `GridSearchCV` with a group-wise inner cross-validation (`GroupKFold`). Models
    without a grid are returned unchanged. The search is not run here: Evaluator fits
    the returned models on its training rows and passes the groups to `fit()`, so the
    inner split never puts rows of one group into both its parts.
    """
    def __init__(self, config_manager: ConfigManager, name: Optional[str] = None) -> None:
        """Reads the tuning settings.

        Args:
            config_manager (ConfigManager): Configuration manager instance.
            name (str | None): Component name.
        """
        super().__init__(config_manager, name)
        self._n_splits: int = int(self.config_manager.get_param("pipeline.tuning.n_splits", 3))
        self._scoring: str = self.config_manager.get_param(
            "pipeline.tuning.scoring", "neg_root_mean_squared_error"
        )
        self._n_jobs: Optional[int] = self.config_manager.get_param("pipeline.tuning.n_jobs")
        self._param_grids: Dict[str, Dict[str, List[Any]]] = (
            self.config_manager.get_param("pipeline.tuning.param_grids", {}) or {}
        )


    def _build_search(self, model: BaseModel) -> Union[BaseModel, GridSearchCV]:
        """Wraps a model into a `GridSearchCV` if it has a parameter grid.

        Args:
            model (BaseModel): The model to tune.

        Returns:
            BaseModel | GridSearchCV: The search, or the unchanged model if it has no grid.

        Raises:
            ValueError: If the grid names a parameter the model does not have or gives
                an empty list of values.
        """
        grid = self._param_grids.get(model.name)
        if not grid:
            return model

        unknown = [param for param in grid if param not in model.get_params(deep=False)] # type: ignore
        if unknown:
            raise ValueError(
                f"Parameters {unknown} of the grid for {model.name}"
                "are not parameters of the model."
            )
        empty = [param for param, values in grid.items() if not values]
        if empty:
            raise ValueError(f"The grid for {model.name} has no values for the parameters {empty}.")

        return GridSearchCV(
            estimator=model,
            param_grid=grid,
            cv=GroupKFold(n_splits=self._n_splits),
            scoring=self._scoring,
            n_jobs=self._n_jobs,
            refit=True,
        )


    def run(
        self,
        models: Sequence[BaseModel],
        *args: Any,
        **kwargs: Any
    ) -> List[Union[BaseModel, GridSearchCV]]:
        """Wraps the models that have a parameter grid into a group-wise grid search.

        Args:
            models (Sequence[BaseModel]): The models to tune.

        Returns:
            List[BaseModel | GridSearchCV]: The models in the same order, ready for
            `Evaluator.run()`.

        Raises:
            ValueError: If the configuration names a model that is not in `models`,
                or a grid is invalid.
        """
        names = {model.name for model in models}
        unused = [name for name in self._param_grids if name not in names]
        if unused:
            raise ValueError(f"Parameter grids are configured for unknown models {unused}.")
        return [self._build_search(model) for model in models]