import pandas as pd
import json
import os
import torch
import shutil
import gc

from helpers import get_data, evaluate
from preprocessing import drop_duplicates, vinai_preprocessing
from models import MODELS_MAPPER

from typing import Any, Dict, List


def conduct_sensitivity_analysis(
    model_class_name: str,
    default_config: Dict[str, Any],
    seed: int,
    train_size: int,
    test_size: int,
    numerical_model_args: List[str],
    factor: float,
    out_path: str
) -> None:
    """
    analyize change of numerical hyperparameters on performance by, for each
    parameter with value v in numerical_model_args, fitting and evaluating a
    model trained with it once set to v * factor and once to v / factor

    :param model_class_name: model class name
    :param default_config: default model arguments
    :param seed: seed for data order
    :param train_size: number of training examples
    :param test_size: number of testing examples
    :param numerical_model_args: numerical model hyperparameters to vary
    :param factor: factor by which to vary parameters
    :param out_path: where to save results
    """
    model_class = MODELS_MAPPER[model_class_name]
    if not os.path.exists(f'{out_path}/probs'):
        os.makedirs(f'{out_path}/probs')
    
    full_config = {
        'data_order_seed': seed,
        'train_data_size': train_size,
        'test_data_size': test_size,
        'numerical_model_args': numerical_model_args,
        'factor': factor,
        **default_config
    }
    with open(f'{out_path}/full_config.json', 'w') as f:
        json.dump(full_config, f)

    X_train, y_train, X_test, y_test = get_data(seed, train_size, test_size)
    X_train, y_train = drop_duplicates(X_train, y_train)
    X_test, y_test = drop_duplicates(X_test, y_test)
    X_train, X_test = vinai_preprocessing(X_train), vinai_preprocessing(X_test)

    configs = {'default': {**default_config}}
    for param in numerical_model_args:
        value = default_config[param]
        neighbors = [(1 / factor) * value, factor * value]
        neighbors = [type(value)(val) for val in neighbors]
        for val in neighbors:
            configs[f'{param}-{val:.2g}'] = {**default_config, param: val}

    torch.save(y_test, f'{out_path}/y_test.pt')
    with open(f'{out_path}/X_test.txt', 'w') as f:
        f.write('\n'.join(X_test))
    names = list(configs.keys())
    metrics = ['acc', 'mcc', 'tp', 'tn', 'fp', 'fn', 'auroc', 'auprc']
    columns = [f'test_{mn}' for mn in metrics]
    results = pd.DataFrame(index=names, columns=columns)
    for name, config in configs.items():
        trained_model = model_class(config)
        trained_model.fit(X_train, y_train)
        probs_out_path = f'{out_path}/probs/{name}.pt'
        res = evaluate(
            trained_model, X_train, y_train, X_test, y_test,
            False, probs_out_path
        )
        for metric_name in metrics:
            idx = f'test_{metric_name}'
            results.loc[name, idx] = res.loc['test', metric_name]
        results.sort_values(by='test_acc', inplace=True, ascending=False)
        results.to_csv(f'{out_path}/results.csv')
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree('outputs')
