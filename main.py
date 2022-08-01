import os
import json
import warnings
from transformers import logging

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
logging.set_verbosity_error()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models import MODELS_MAPPER
from trials import TRIALS_MAPPER
from eda import conduct_frequencies_eda
from helpers import fit_and_evaluate_baseline_models
from sensitivity_analysis import conduct_sensitivity_analysis
from ensembling import (
    ensembling_candidate_selection,
    fit_ensemble_models,
    select_final_submission
)


def main(
    run_exploratory_data_analysis=False,
    run_baselines=False,
    run_sensitivity_analysis=False,
    run_ensembling_candidate_search=False,
    run_full_ensemble=False
) -> None:
    """
    perform whichever section of the process indicated by True

    :param run_exploratory_data_analysis: whether to run EDA
    :param run_baselines: whether to train baseline models
    :param run_sensitivity_analysis: whether to do sensitivity analysis
    :param run_ensembling_candidate_search: whether to run candidate search
    :param run_full_ensemble: whether to train ensemble candidates
    """
    with open('config.json', 'r') as f:
        config = json.load(f)

    if run_exploratory_data_analysis:
        conduct_frequencies_eda(**config['exploratory_data_analysis'])
    if run_baselines:
        fit_and_evaluate_baseline_models(
            **config['baselines'],
            seed=config['seed'],
            test_size=config['default_test_size'],
            models_mapper=MODELS_MAPPER
        )
    if run_sensitivity_analysis:
        conduct_sensitivity_analysis(
            **config['sensitivity_analysis'],
            test_size=config['default_test_size'],
            model_class_name=config['default_model_class'],
            default_config=config['default_model_args'],
            seed=config['seed']
        )
    if run_ensembling_candidate_search:
        common_ensembling_search_args = {
            **config['ensembling_candidate_search']['common'],
            'model_class_name': config['default_model_class'],
            'default_config': config['default_model_args'],
            'seed': config['seed'],
            'test_size': config['default_test_size'],
            'proxy_train_size': config['proxy_train_size']
        }
        ensembling_candidate_selection(
            **config['ensembling_candidate_search']['subset_curriculum'],
            **common_ensembling_search_args
        )
        ensembling_candidate_selection(
            **config['ensembling_candidate_search']['variance_benchmark'],
            **common_ensembling_search_args
        )
    if run_full_ensemble:
        fit_ensemble_models(
            **config['full_ensemble'],
            data_seed=config['seed'],
            train_size=config['full_train_size'],
            test_size=config['default_test_size'],
            proxy_train_size=config['proxy_train_size'],
            proxy_model_class=config['default_model_class'],
            proxy_model_args=config['default_model_args'],
            component_model_class=config['default_model_class'],
            component_model_config=config['default_model_args'],
            component_candidates=TRIALS_MAPPER['ensembling_candidate_trials'],
        )
        select_final_submission(
            out_path=config['full_ensemble']['out_path']
        )


if __name__ == '__main__':
    main(
        run_exploratory_data_analysis=False,
        run_baselines=False,
        run_sensitivity_analysis=False,
        run_ensembling_candidate_search=False,
        run_full_ensemble=False
    )
