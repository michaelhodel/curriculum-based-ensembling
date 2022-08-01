import tqdm
import pandas as pd
import torch
import json
import os
import gc
import random
import numpy as np
import itertools
from copy import deepcopy

from typing import Tuple, Dict, List, Callable, Any

from helpers import get_data, get_holdout, shuffle, calc_metrics, evaluate
from trials import run_trials
from models import MODELS_MAPPER
import preprocessing
import curriculum


def ensemble_via_label_mode(
    probabilities: pd.DataFrame
) -> np.array:
    """
    compute mode of predicted labels

    :param probabilities: predicted probabilities for positive labels
    :return: ensemble probabilities
    """
    predicted_labels = probabilities.round()
    ensembled_probabilities = predicted_labels.mean(axis=1).round()
    return ensembled_probabilities.values


def ensemble_via_arithmetic_mean_probability(
    probabilities: pd.DataFrame
) -> np.array:
    """
    compute arithmetic mean of probabilities

    :param probabilities: predicted probabilities for positive labels
    :return: ensemble probabilities
    """
    ensembled_probabilities = probabilities.mean(axis=1)
    return ensembled_probabilities.values


def ensemble_via_geometric_mean_odds(
    probabilities: pd.DataFrame
) -> np.array:
    """
    compute geometric mean of odds
    
    :param probabilities: predicted probabilities for positive labels
    :return: ensemble probabilities
    """
    n_components = probabilities.shape[1]
    odds = probabilities.divide(1 - probabilities)
    mean_odds = odds.product(axis=1) ** (1 / n_components)
    ensembled_probabilities = (mean_odds / (1 + mean_odds))
    return ensembled_probabilities.values


def ensemble_via_maximum_confidence(
    probabilities: pd.DataFrame
) -> np.array:
    """
    compute most confidence probabilities

    :param probabilities: predicted probabilities for positive labels
    :return: ensemble probabilities
    """
    get_max_confidence = lambda x: x[(x - 0.5).abs().idxmax()]
    ensembled_probabilities = probabilities.apply(get_max_confidence, axis=1)
    return ensembled_probabilities.values


def ensemble_probabilities(
    probabilities: pd.DataFrame,
    inference_style: str
) -> np.array:
    """
    makes probabilities from an ensemble of probabilities

    :param probabilities: a dict of predicted probabilities
    :param inference_style: function identifier
    """
    inference_style_mapper = {
        'pred_mode': ensemble_via_label_mode,
        'prob_mean_arith': ensemble_via_arithmetic_mean_probability,
        'odds_mean_geom': ensemble_via_geometric_mean_odds,
        'conf_max': ensemble_via_maximum_confidence
    }
    return inference_style_mapper[inference_style](probabilities)


class Ensemble:
    """
    ensemble of classifiers
    """
    def __init__(
        self,
        model_configs,
        vary_data_order: bool,
        vary_weight_init: bool,
        inference_style: str,
        n_models=None,
    ) -> None:
        """
        initialization

        :param model_configs: either a config dict or list of config dicts
        :param vary_data_order: whether to vary data order during training
        :param vary_weight_init: whether to vary RNG for weight initialization
        :param inference_style: identifier for inference function
        :param n_models: number of models, if model_configs not an iterable
        """
        if isinstance(model_configs, dict):
            model_class = model_configs['model_class']
            model_args = model_configs['model_args']
            self.model_classes = [model_class for _ in range(n_models)]
            self.model_args = [model_args.copy() for _ in range(n_models)]
        else:
            self.model_classes = [c['model_class'] for c in model_configs]
            self.model_args = [c['model_args'].copy() for c in model_configs]
        self.n_models = len(self.model_classes)
        self.vary_data_order = vary_data_order
        self.vary_weight_init = vary_weight_init
        self.inference_style = inference_style
        self.clfs = {}

    def fit(
        self,
        X: Tuple[str],
        y: torch.Tensor
    ) -> None:
        """
        train ensemble of models

        :param X: tweets
        :param y: labels
        """
        random.seed(42)
        seeds = random.sample(range(69), self.n_models)
        for i, seed in enumerate(seeds):
            if self.vary_data_order:
                X, y = shuffle(X, y, seed)
            if self.vary_weight_init:
                self.model_args[i] = {
                    **self.model_args[i], 'manual_seed': seed
                }
            clf = self.model_classes[i](self.model_args[i])
            clf.fit(X, y)
            self.clfs[i] = clf

    def predict_proba(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        """
        predict probabilities

        :param X: tweets
        :return: predicted probabilities
        """
        probs = {}
        for i, clf in self.clfs.items():
            probs[i] = clf.predict_proba(X).numpy()[:, 1]
        probs = pd.DataFrame(probs)
        probs = ensemble_probabilities(probs, self.inference_style)
        probs = np.stack((1 - probs, probs), axis=1)
        return torch.Tensor(probs)

    def predict(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        """
        predict labels

        :param X: tweets
        :return: predicted labels
        """
        return 2 * self.predict_proba(X).round() - 1


def ensembling_candidates_assessment(
    out_path: str,
    n_models_list: List[int],
    inference_styles: List[str]
) -> None:
    """
    grid-search ensembles via evaluating each possible ensemble from
    all combinations of n models with inference style inference_stlye
    for each n in n_models_list and each inference_style in inference_styles

    :param out_path: path of experiment results, with data and probabilities
    :param n_models_list: list number of models per ensemble
    :param inference_styles: list of inference styles
    """
    results = pd.read_csv(f'{out_path}/results.csv', index_col=0)
    y_test = torch.load(f'{out_path}/y_test.pt')
    candidates = results.index.tolist()
    probs = {
        candidate: torch.load(
            f'{out_path}/probs/{candidate}.pt'
        ).numpy()[:, 1] for candidate in candidates
    }
    metrics = ['acc', 'mcc', 'tp', 'tn', 'fp', 'fn', 'auroc', 'auprc']
    candidate_search_path = f'{out_path}/candidate_search'
    if not os.path.exists(candidate_search_path):
        os.makedirs(candidate_search_path)
    for n_models in n_models_list:
        model_path = f'{candidate_search_path}/ensemble_{n_models}'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        combinations = list(itertools.combinations(candidates, n_models))
        for inference_style in inference_styles:
            columns = [f'model {i}' for i in range(1, n_models + 1)] + metrics
            candidate_search_results = pd.DataFrame(
                index=range(len(combinations)), columns=columns
            )
            pbar = tqdm.tqdm(enumerate(combinations), total=len(combinations))
            for i, combination in pbar:
                component_probs = pd.DataFrame(
                    {model_id: probs[model_id] for model_id in combination}
                )
                ensembled_probs = ensemble_probabilities(
                    component_probs, inference_style
                )
                test_probs = torch.Tensor(
                    np.stack((1 - ensembled_probs, ensembled_probs), axis=1)
                )
                metrics_res = calc_metrics(y_test, test_probs)
                for j in range(n_models):
                    idx = f'model {j + 1}'
                    candidate_search_results.loc[i, idx] = combination[j]
                for m in metrics:
                    candidate_search_results.loc[i, m] = metrics_res[m]
            candidate_search_results.sort_values(
                by='acc', ascending=False, inplace=True
            )
            path = f'{model_path}/{inference_style}.csv'
            candidate_search_results.to_csv(path)
    

def choose_ensembling_components(
    n_models_list: List[int],
    inference_styles: List[str],
    out_path: str,
    pct: float,
) -> None:
    """
    determines n ensemble components by choosing the n - 1 ensemble candidates
    that occur most often in the top performing pct of models, together
    with the candidate that most often occurs with any n - 2 of the
    most occurring n - 1 candidates, also in the top models,
    for each n in n_models_list and each inference_style in inference_styles

    :param n_models_list: list number of models per ensemble
    :param inference_styles: list of inference styles
    :param out_path: where to save results to
    :param pct: indicating which share of best models to consider
    """
    candidates_search_summary = pd.DataFrame(
        index=n_models_list, columns=inference_styles
    )
    best_ensembles_summary = dict()
    for n_models in n_models_list:
        ensemble_path = f'{out_path}/candidate_search/ensemble_{n_models}'
        best_ensembles_summary[n_models] = dict()
        pbar = tqdm.tqdm(inference_styles, total=len(inference_styles))
        for inference_style in pbar:
            path = f'{ensemble_path}/{inference_style}.csv'
            df = pd.read_csv(path, index_col=0)
            df.sort_values(by='acc', ascending=True)
            k = df.shape[0]
            model_cols = [c for c in df.columns if 'model ' in c]
            for c in model_cols:
                df[c] = df[c].astype(str)
            top_models = df[:int(k * pct)][model_cols]
            ensembles = set(top_models.to_numpy().flatten())
            ensemble_subsets = list(
                itertools.combinations(ensembles, n_models - 1)
            )
            ensemble_subsets = [
                tuple(sorted(subset)) for subset in ensemble_subsets
            ]
            frequencies = {subset: 0 for subset in ensemble_subsets}
            for i, models in top_models.iterrows():
                subset_occurrences = list(
                    itertools.combinations(models.tolist(), n_models - 1)
                )
                for subset in subset_occurrences:
                    frequencies[tuple(sorted(subset))] += 1
            frequencies = sorted(
                frequencies.items(), key=lambda t: t[1], reverse=True
            )
            best_components = set(frequencies[0][0])
            last_component_candidates = ensembles - best_components
            last_component_candidate_scores = {
                candidate: 0 for candidate in last_component_candidates
            }
            subsets = list(
                itertools.combinations(best_components, n_models - 2)
            )
            subsets = [set(subset) for subset in subsets]
            for i, models in top_models.iterrows():
                models_set = set(models.tolist())
                for subset in subsets:
                    if subset < models_set:
                        for candidate in models_set - best_components:
                            last_component_candidate_scores[candidate] += 1
            last_component_candidate_scores = sorted(
                last_component_candidate_scores.items(),
                key=lambda t: t[1], reverse=True
            )
            best_last_candidate = last_component_candidate_scores[0][0]
            best_components.add(best_last_candidate)
            best_components = sorted(list(best_components))
            to_set = lambda c: set(c.tolist())
            idx = df[model_cols].apply(to_set, axis=1) == set(best_components)
            best_ensemble_score = df[idx]['acc'].item()
            best_ensembles_summary[n_models][inference_style] = {
                'components': best_components, 'accuracy': best_ensemble_score
            }
            top_score = df[:int(k * pct)]['acc'].mean()
            candidates_search_summary.loc[n_models, inference_style] = top_score
    candidates_search_summary.to_csv(f'{out_path}/candidate_search_summary.csv')
    with open(f'{out_path}/best_ensembles_summary.json', 'w') as f:
        json.dump(best_ensembles_summary, f)


def ensembling_candidate_selection(
    trials_name: str,
    model_class_name: str,
    default_config: Dict[str, Any],
    seed: int,
    train_size: int,
    test_size: int,
    proxy_train_size: int,
    eval_on_train: bool,
    out_path: str,
    n_models_list: List[int],
    inference_styles: List[str],
    pct: float,
) -> None:
    """
    runs trials, assesses them as ensembling candidates and picks best

    :param trials_name: name of trials
    :param model_class_name: model class name
    :param default_config: default model args
    :param seed: seed for data order
    :param train_size: number of training examples
    :param test_size: number of testing examples
    :param proxy_train_size: number of examples for proxy model
    :param eval_on_train: whether to evaluate on training data
    :param out_path: where to save results to
    :param n_models_list: list number of models per ensemble
    :param inference_styles: list of inference styles
    :param pct: indicating which share of best models to consider
    """
    run_trials(
        trials_name, model_class_name, default_config, seed,
        train_size, test_size, proxy_train_size, eval_on_train, out_path
    )
    ensembling_candidates_assessment(out_path, n_models_list, inference_styles)
    choose_ensembling_components(n_models_list, inference_styles, out_path, pct)


def get_ensemble_components(
    component_candidates: Dict[str, Callable],
    ensemble_search_path: str
) -> Dict[str, Callable]:
    """
    select the ensemble component model fitting functions
    according to the selection procedure

    :param component_candidates: candidates over which search was performed
    :param ensemble_search_path: path of search results
    :return: dictionary of (component_id, component_callable)
    """
    with open(f'{ensemble_search_path}/best_ensembles_summary.json', 'r') as f:
        best_ensembles_summary = json.load(f)
    n = max(best_ensembles_summary.keys())
    best_acc = 0
    best_components = []
    for inference_method, res in best_ensembles_summary[n].items():
        if res['accuracy'] > best_acc:
            best_acc = res['accuracy']
            best_components = res['components']
    components = {}
    for component in best_components:
        components[component] = component_candidates[component]
    return components


def fit_ensemble_models(
    out_path: str,
    unique_tweets_only: bool,
    no_spam_tweets: bool,
    standard_preprocessing: bool,
    save_models: bool,
    data_seed: int,
    train_size: int,
    proxy_train_size: int,
    test_size: int,
    proxy_model_class: Any,
    proxy_model_args: Any,
    component_model_class: Any,
    component_model_config: Any,
    component_candidates: Dict[str, Callable],
    ensemble_search_path: str,
) -> None:
    """
    fit ensemble component models and save relevant outputs

    :param out_path: where to save results
    :param unique_tweets_only: whether to delete duplicate tweets
    :param no_spam_tweets: whether to filter out spam tweets
    :param standard_preprocessing: whether to apply vinai preprocessing
    :param save_models: whether to save models
    :param data_seed: seed for determining initial data oder
    :param train_size: number of training examples for ensemble components
    :param proxy_train_size: number of training examples for proxy model
    :param test_size: number of examples for internal test set
    :param proxy_model_class: proxy model class
    :param proxy_model_args: proxy model args
    :param component_model_class: component model class
    :param component_model_config: component model args
    :param component_candidates: component candidates
    :param ensemble_search_path: path to ensemble search outputs
    """
    if os.path.exists(out_path):
        raise AssertionError(
            f'"{out_path}" already exists, choose another path!'
        )
    os.makedirs(out_path)

    X_train, y_train, X_test, y_test = get_data(
        data_seed, train_size + proxy_train_size, test_size
    )

    X_proxy, y_proxy = X_train[train_size:], y_train[train_size:]
    X_train, y_train = X_train[:train_size], y_train[:train_size]
    X = get_holdout()

    if unique_tweets_only:
        X_proxy, y_proxy = preprocessing.drop_duplicates(X_proxy, y_proxy)
        X_train, y_train = preprocessing.drop_duplicates(X_train, y_train)
        X_test, y_test = preprocessing.drop_duplicates(X_test, y_test)

    torch.save(y_test, f'{out_path}/y_test.pt')
    with open(f'{out_path}/X_test.txt', 'w') as f:
        f.write('\n'.join(X_test))

    if no_spam_tweets:
        X_proxy, y_proxy = preprocessing.filter_spam(X_proxy, y_proxy, True)
        X_train, y_train = preprocessing.filter_spam(X_train, y_train, True)

    if standard_preprocessing:
        X_proxy = preprocessing.vinai_preprocessing(X_proxy)
        X_train = preprocessing.vinai_preprocessing(X_train)
        X_test = preprocessing.vinai_preprocessing(X_test)
        X = preprocessing.vinai_preprocessing(X)

    proxy_model_class = MODELS_MAPPER[proxy_model_class]
    component_model_class = MODELS_MAPPER[component_model_class]

    ensemble_components = get_ensemble_components(
        component_candidates=component_candidates,
        ensemble_search_path=ensemble_search_path
    )

    print(f'training proxy model for estimating difficulties:')
    proxy_model = proxy_model_class(proxy_model_args)
    proxy_model.fit(X_proxy, y_proxy)
    probs = proxy_model.predict_proba(X_train)
    print('estimating difficulties:')
    errors = curriculum.get_errors(y_train, probs)
    torch.save(torch.Tensor(errors), f'{out_path}/errors.pt')

    n = len(ensemble_components)
    random.seed(data_seed)
    seeds = random.sample(range(69), n)
    iterable = enumerate(ensemble_components.items())
    for i, (model_name, model_callable) in iterable:
        model_path = f'{out_path}/{model_name}'
        print(f'fitting model {i + 1} out of {n} ({model_name}):')
        os.makedirs(model_path)

        seed = seeds[i]
        X_train_shuffled, y_train_shuffled = shuffle(
            deepcopy(X_train), y_train.clone(), seed
        )
        config = {**component_model_config, 'manual_seed': seed}
        if save_models:
            config['output_dir'] = f'{model_path}/outputs'
        clf = model_callable(
            model_class=component_model_class,
            default_config=config,
            X=X_train_shuffled,
            y=y_train_shuffled,
            errors=errors
        )

        probabilities = clf.predict_proba(X)
        predictions = 2 * probabilities[:, 1].round() - 1
        predictions = predictions.to(int).tolist()
        ids = range(1, len(X) + 1)
        submission = pd.DataFrame(
            [ids, predictions], index=['Id', 'Prediction']
        )
        submission.T.to_csv(f'{model_path}/submission.csv', index=False)
        torch.save(probabilities, f'{model_path}/probabilities.pt')

        res = evaluate(
            clf, X_train, y_train, X_test, y_test,
            False, f'{model_path}/test_probabilities.pt'
        )
        res.to_csv(f'{model_path}/test_results.csv')

        component_config = {
            'data_order_seed': seed,
            'component_type': model_name,
            **config
        }
        with open(f'{model_path}/config.json', 'w') as f:
            json.dump(component_config, f)

        gc.collect()
        torch.cuda.empty_cache()


def select_final_submission(
    out_path: str
) -> None:
    """
    selects best ensemble for submission

    :param out_path: path to folder containing outputs from full ensembles
    """
    test_results = pd.DataFrame()
    components = []
    for fn in os.listdir(out_path):
        if os.path.isdir(f'{out_path}/{fn}'):
            if 'probabilities.pt' in os.listdir(f'{out_path}/{fn}'):
                components.append(fn)
    for component in components:
        res_path = f'{out_path}/{component}/test_results.csv'
        res = pd.read_csv(res_path, index_col=0)
        res.index.values[0] = component
        test_results = pd.concat((test_results, res))
    test_results.sort_values(by='acc', ascending=False, inplace=True)
    test_results.to_csv(f'{out_path}/results.csv')

    candidates = test_results.index.tolist()
    probs_dir = f'{out_path}/probs'
    if not os.path.exists(probs_dir):
        os.makedirs(probs_dir)
    for candidate in candidates:
        probs = torch.load(f'{out_path}/{candidate}/test_probabilities.pt')
        torch.save(probs, f'{out_path}/probs/{candidate}.pt')
    n_models_list = list(range(2, len(candidates) + 1))
    inference_styles = [
        'pred_mode', 'prob_mean_arith', 'odds_mean_geom', 'conf_max'
    ]
    ensembling_candidates_assessment(
        out_path=out_path,
        n_models_list=n_models_list,
        inference_styles=inference_styles
    )
    choose_ensembling_components(
        n_models_list=n_models_list,
        inference_styles=inference_styles,
        out_path=out_path,
        pct=1
    )
    with open(f'{out_path}/best_ensembles_summary.json', 'r') as f:
        best_ensembles_summary = json.load(f)
    best_acc = 0
    best_components = []
    best_inference_method = ''
    for n in best_ensembles_summary.keys():
        for inference_method, res in best_ensembles_summary[n].items():
            if res['accuracy'] > best_acc:
                best_acc = res['accuracy']
                best_components = res['components']
                best_inference_method = inference_method

    selection = {
        'components': best_components,
        'inference_method': best_inference_method,
        'test_set_accuracy': best_acc
    }
    with open(f'{out_path}/final_selection.json', 'w') as f:
        json.dump(selection, f)

    holdout_probs = {
        component: torch.load(
            f'{out_path}/{component}/probabilities.pt'
        ).numpy()[:, 1] for component in best_components
    }

    holdout_probs = pd.DataFrame(holdout_probs)
    probabilities = ensemble_probabilities(
        probabilities=holdout_probs,
        inference_style=best_inference_method
    )
    predictions = (2 * probabilities.round() - 1).astype(int).tolist()
    ids = range(1, len(predictions) + 1)
    submission = pd.DataFrame([ids, predictions], index=['Id', 'Prediction'])
    submission.T.to_csv(f'{out_path}/submission.csv', index=False)


