import os
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import shap
from shap import KernelExplainer as Explainer

from predict import predict_and_rescale
from utils import read_json_as_dict

EXPLAINER_FILE_NAME = "explainer.joblib"


class RegressionExplainer:
    """Shap Explainer class for regression models"""

    EXPLANATION_METHOD = "Shap"

    def __init__(self, max_local_explanations: int = 5):
        self.max_local_explanations = max_local_explanations
        self._shap_explainer = None
        self._explainer_data = None

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit the explainer to the training data.

        Args:
            train_data (pd.DataFrame): Training data to use for explainer.
        """
        # rather than use the whole training set to estimate expected values,
        # we summarize with a set of weighted kmeans, each weighted by the number
        # of points they represent.
        self._explainer_data = shap.kmeans(train_data, 10)

    def _build_explainer(self, predictor_model, target_encoder):
        """Build shap explainer

        Args:
            predictor_model (Any): A trained predictor model

        Returns:
            'Explainer': instance of shap Explainer from shap library
        """
        return Explainer(
            model=lambda instances: predict_and_rescale(
                predictor_model, target_encoder, instances
            ),
            data=self._explainer_data,
            seed=0,
        )

    def get_explanations(
        self,
        instances_df: pd.DataFrame,
        predictor_model: Any,
        target_encoder: Any,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get local explanations for the given instances.

        Args:
            instances_df (pd.DataFrame): Instances to explain predictions
            predictor_model (Any): A trained predictor model
            target_encoder (Any): A trained target encoder

        Returns:
            Dict[str, Any]: Explanations returned in a dictionary
        """
        # limit explanations to at most self.max_local_explanations
        instances_df = instances_df.head(self.max_local_explanations)
        if self._shap_explainer is None:
            self._shap_explainer = self._build_explainer(
                predictor_model, target_encoder
            )
        explanations = []
        shap_values = self._shap_explainer.shap_values(instances_df)[0]
        # indexing at 0 because this returns a list of explanations for each "class".
        # For regression and binary classification, we will always have just one
        # item in this list. For multiclass-classification, we will have k-elements
        # where k is number of classes.
        baseline = self._shap_explainer.expected_value
        for row_num in range(len(instances_df)):
            feature_scores = {}
            for f_num, feature in enumerate(instances_df.columns):
                feature_scores[feature] = round(shap_values[row_num][f_num], 4)
            explanations.append(
                {
                    "baseline": round(baseline[0], 4),
                    "featureScores": feature_scores,
                }
            )

        return {
            "explanation_method": self.EXPLANATION_METHOD,
            "explanations": explanations,
        }

    def save(self, file_path: Path) -> None:
        """Save the explainer to a file."""
        with open(file_path, "wb") as file:
            joblib.dump(self, file)

    @classmethod
    def load(cls, file_path: Path) -> "RegressionExplainer":
        """Load the explainer from a file."""
        with open(file_path, "rb") as file:
            loaded_explainer = joblib.load(file)
        return loaded_explainer


def fit_and_save_explainer(
    train_data: pd.DataFrame, explainer_config_file_path: str, save_dir_path: str
) -> None:
    """
    Fit the explainer to the training data and save it to a file.

    Args:
        train_data (pd.DataFrame): pandas DataFrame of training data
        explainer_config_file_path (str): Path to the explainer configuration file.
        save_dir_path (str): Dir path where explainer should be saved.

    Returns:
        RegressionExplainer: Instance of RegressionExplainer
    """
    explainer_config = read_json_as_dict(explainer_config_file_path)
    explainer = RegressionExplainer(
        max_local_explanations=explainer_config["max_local_explanations"]
    )
    explainer.fit(train_data)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    explainer.save(os.path.join(save_dir_path, EXPLAINER_FILE_NAME))
    return explainer


def load_explainer(save_dir_path: str) -> Any:
    """
    Load the explainer from a file.

    Args:
        save_dir_path (str): Dir path where explainer is saved.

    Returns:
        RegressionExplainer: Instance of RegressionExplainer
    """
    return RegressionExplainer.load(os.path.join(save_dir_path, EXPLAINER_FILE_NAME))


def get_explanations_from_explainer(
    instances_df: pd.DataFrame,
    explainer: RegressionExplainer,
    predictor_model: Any,
    target_encoder: Any,
) -> Dict[str, Any]:
    """Get explanations for the given instances_df.

    Args:
        instances_df (pd.DataFrame): instances to explain predictions
        explainer (RegressionExplainer): Instance of
                    RegressionExplainer
        predictor_model (Any): A trained predictor model
        target_encoder (Any): A trained target encoder

    Returns:
        Dict[str, Any]: Explanations returned in a dictionary
    """
    return explainer.get_explanations(instances_df, predictor_model, target_encoder)
