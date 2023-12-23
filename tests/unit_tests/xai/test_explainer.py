import os
from typing import Any

import pytest
from pandas import DataFrame
from py.path import local as LocalPath

from xai.explainer import (
    RegressionExplainer,
    fit_and_save_explainer,
    get_explanations_from_explainer,
    load_explainer,
)


def test_fit_explainer(transformed_train_inputs: DataFrame) -> None:
    """
    Test fitting of the explainer.

    Args:
        transformed_train_inputs (DataFrame): Transformed train inputs.
    """
    explainer = RegressionExplainer()
    explainer.fit(transformed_train_inputs)
    assert explainer._explainer_data is not None


def test_build_explainer(
    transformed_train_inputs: DataFrame, predictor: Any, target_encoder: Any
) -> None:
    """
    Test building of the explainer.

    Args:
        transformed_train_inputs (DataFrame): Transformed train inputs.
        predictor (Any): A predictor model object.
        target_encoder (Any): Target encoder.
    """
    explainer = RegressionExplainer()
    explainer.fit(transformed_train_inputs)
    shap_explainer = explainer._build_explainer(predictor, target_encoder)
    assert shap_explainer is not None


@pytest.mark.slow
def test_get_explanations(
    transformed_test_inputs: DataFrame, predictor: Any, target_encoder: Any
) -> None:
    """
    Test getting explanations.

    Args:
        transformed_test_inputs (DataFrame): Transformed test inputs.
        predictor (Any): A predictor model object.
        target_encoder (Any): Target encoder.

    """
    explainer = RegressionExplainer()
    explainer.fit(transformed_test_inputs)
    explanations = explainer.get_explanations(
        transformed_test_inputs, predictor, target_encoder
    )
    assert explanations is not None
    assert "explanation_method" in explanations
    assert "explanations" in explanations


def test_save_and_load_explainer(
    tmpdir: LocalPath, transformed_train_inputs: DataFrame
) -> None:
    """
    Test saving and loading of the explainer.

    Args:
        tmpdir (LocalPath): Temporary directory path provided by pytest.
        transformed_train_inputs (DataFrame): Transformed train inputs.

    """
    explainer = RegressionExplainer(max_local_explanations=10)
    explainer.fit(transformed_train_inputs)
    explainer_dir_path = tmpdir.join("explainer")
    explainer.save(explainer_dir_path)
    loaded_explainer = RegressionExplainer.load(explainer_dir_path)
    assert loaded_explainer is not None
    assert loaded_explainer._explainer_data is not None
    assert loaded_explainer.max_local_explanations == 10


def test_fit_and_save_explainer(
    transformed_train_inputs: DataFrame,
    config_file_paths_dict: dict,
    resources_paths_dict: dict,
) -> None:
    """
    Test fitting and saving of the explainer.

    Args:
        transformed_train_inputs (DataFrame): Transformed train inputs.
        config_file_paths_dict (dict): Dictionary containing the paths to the
            configuration files.
        resources_paths_dict (dict): Dictionary containing the paths to the
            resources files such as trained models, encoders, and explainers.

    """
    explainer_config_file_path = config_file_paths_dict["explainer_config_file_path"]
    explainer_dir_path = resources_paths_dict["explainer_dir_path"]
    explainer = fit_and_save_explainer(
        transformed_train_inputs, explainer_config_file_path, explainer_dir_path
    )
    assert len(os.listdir(explainer_dir_path)) == 1
    assert explainer is not None
    assert explainer._explainer_data is not None


def test_load_explainer(
    transformed_train_inputs: DataFrame,
    config_file_paths_dict: dict,
    resources_paths_dict: dict,
) -> None:
    """
    Test loading of the explainer.

    Args:
        transformed_train_inputs (DataFrame): Transformed train inputs.
        config_file_paths_dict (dict): Dictionary containing the paths to the
            configuration files.
        resources_paths_dict (dict): Dictionary containing the paths to the
            resources files such as trained models, encoders, and explainers.

    """
    explainer_config_file_path = config_file_paths_dict["explainer_config_file_path"]
    explainer_dir_path = resources_paths_dict["explainer_dir_path"]
    _ = fit_and_save_explainer(
        transformed_train_inputs, explainer_config_file_path, explainer_dir_path
    )
    loaded_explainer = load_explainer(explainer_dir_path)
    assert loaded_explainer is not None
    assert loaded_explainer._explainer_data is not None


@pytest.mark.slow
def test_get_explanations_from_explainer(
    transformed_test_inputs: DataFrame,
    config_file_paths_dict: dict,
    resources_paths_dict: dict,
    predictor: Any,
    target_encoder: Any,
) -> None:
    """
    Test the test_get_explanations_from_explainer function.

    Args:
        transformed_test_inputs (DataFrame): Transformed test inputs.
        config_file_paths_dict (dict): Dictionary containing the paths to the
            configuration files.
        resources_paths_dict (dict): Dictionary containing the paths to the
            resources files such as trained models, encoders, and explainers.
        predictor (Any): A predictor model object.
        target_encoder (Any): Target encoder.

    """
    explainer_config_file_path = config_file_paths_dict["explainer_config_file_path"]
    explainer_dir_path = resources_paths_dict["explainer_dir_path"]
    explainer = fit_and_save_explainer(
        transformed_test_inputs, explainer_config_file_path, explainer_dir_path
    )
    explanations = get_explanations_from_explainer(
        transformed_test_inputs, explainer, predictor, target_encoder
    )
    assert explanations is not None
    assert "explanation_method" in explanations
    assert "explanations" in explanations
    assert isinstance(explanations["explanations"], list)
    for explanation in explanations["explanations"]:
        assert isinstance(explanation, dict)
        assert "baseline" in explanation
        assert "featureScores" in explanation


@pytest.mark.slow
def test_explanations_from_loaded_explainer(
    transformed_train_inputs: DataFrame,
    config_file_paths_dict: dict,
    resources_paths_dict: dict,
    predictor: Any,
    transformed_test_inputs: DataFrame,
    target_encoder: Any,
) -> None:
    """
    Test loading of the explainer and getting explanations.

    Args:
        transformed_train_inputs (DataFrame): Transformed train inputs.
        config_file_paths_dict (dict): Dictionary containing the paths to the
            configuration files.
        resources_paths_dict (dict): Dictionary containing the paths to the
            resources files such as trained models, encoders, and explainers.
        predictor (Any): A predictor model object.
        transformed_test_inputs (DataFrame): Transformed test inputs.
        target_encoder (Any): Target encoder.

    """
    explainer_config_file_path = config_file_paths_dict["explainer_config_file_path"]
    explainer_dir_path = resources_paths_dict["explainer_dir_path"]
    fit_and_save_explainer(
        transformed_train_inputs, explainer_config_file_path, explainer_dir_path
    )
    loaded_explainer = load_explainer(explainer_dir_path)
    assert loaded_explainer._explainer_data is not None

    explanations = get_explanations_from_explainer(
        transformed_test_inputs, loaded_explainer, predictor, target_encoder
    )
    assert explanations is not None
