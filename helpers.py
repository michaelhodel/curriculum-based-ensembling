import urllib.request
import zipfile
import json
import io
import os
import random
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    auc,
    average_precision_score
)

from typing import Tuple, Dict, Any


def download_data(
    data_url: str = 'http://www.da.inf.ethz.ch/files/twitter-datasets.zip',
    data_fname: str = 'twitter-datasets.zip'
) -> None:
    """
    downlaods data if not stored on disk already

    :param data_url: where to download data from
    :param data_fname: where to store data to
    """
    if not os.path.exists(data_fname):
        urllib.request.urlretrieve(data_url, data_fname)


def get_data(
    seed: int,
    train_size: int,
    test_size: int
) -> Tuple[Tuple[str], torch.Tensor, Tuple[str], torch.Tensor]:
    """
    loads training and testing data, shuffled according to seed,
    with an equal number of positive and negative examples

    :param seed: a seed to determine the shuffling
    :param train_size: the absolute size of the train set
    :param test_size: the absolute size of the test set
    :return: a tuple (X_train, y_train, X_test, y_test)
    """
    download_data()
    suffix = '_full' if train_size + test_size > 2e5 else ''
    class_size = (train_size + test_size) // 2

    neg_path = f'twitter-datasets/train_neg{suffix}.txt'
    pos_path = f'twitter-datasets/train_pos{suffix}.txt'

    with zipfile.ZipFile('twitter-datasets.zip') as zf:
        with io.TextIOWrapper(zf.open(neg_path), encoding='utf-8') as f:
            neg = f.read().split('\n')[:-1][:class_size]
        with io.TextIOWrapper(zf.open(pos_path), encoding='utf-8') as f:
            pos = f.read().split('\n')[:-1][:class_size]
    
    data = list(zip(neg, [-1] * len(neg))) + list(zip(pos, [1] * len(pos)))
    random.Random(seed).shuffle(data)
    train, test = data[:train_size], data[train_size:]

    X_train, y_train = list(zip(*train))
    X_test, y_test = list(zip(*test))
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)

    return X_train, y_train, X_test, y_test


def calc_metrics(
    sentiments: torch.Tensor,
    probs: torch.Tensor
) -> Dict[str, float]:
    """
    compute various metrics to assess performance
    directly inspired by simpletransformers.ClassificationModel.compute_metrics

    :param sentiments: classes
    :param probs: predited probabilities
    """
    labels = sentiments.numpy() / 2 + 0.5
    preds = np.argmax(probs.numpy(), axis=1)
    pos_probs = probs.numpy()[:, 1]
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr, tpr, _ = roc_curve(labels, pos_probs)
    metrics = {
        'acc': accuracy_score(labels, preds),
        'mcc': matthews_corrcoef(labels, preds),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'auroc': auc(fpr, tpr),
        'auprc': average_precision_score(labels, pos_probs)
    }
    return metrics


def evaluate(
    model: Any,
    X_train: Tuple[str],
    y_train: torch.Tensor,
    X_test: Tuple[str],
    y_test: torch.Tensor,
    eval_on_train: bool = False,
    probs_path: str = None
) -> pd.DataFrame:
    """
    evaluate model on (train and) test set

    :param model: a model
    :param X_train: training tweets
    :param y_train: training labels
    :param X_test: testing tweets
    :param y_test: testing labels
    :param eval_on_train: whether to evaluate on train set
    :param probs_path: where to save test probabilities to, if not None
    """
    res_list = []
    if eval_on_train:
        train_probs = model.predict_proba(X_train)
        train_res = calc_metrics(y_train, train_probs)
        res_list.append(('train', train_res))
    test_probs = model.predict_proba(X_test)
    test_res = calc_metrics(y_test, test_probs)
    res_list.append(('test', test_res))
    metrics = ['acc', 'mcc', 'tp', 'tn', 'fp', 'fn', 'auroc', 'auprc']
    results = pd.DataFrame(index=[tup[0] for tup in res_list], columns=metrics)
    for traintest, res in res_list:
        for metric in metrics:
            results.loc[traintest, metric] = res[metric]
    if probs_path is not None:
        torch.save(test_probs, probs_path)
    return results


def get_holdout(
    path: str = 'twitter-datasets/test_data.txt',
    fname: str = 'twitter-datasets.zip'
) -> Tuple[str]:
    """
    get holdout set for submissions

    :param path: where holdout set is located
    :param fname: name of file containing holdout set
    :return: tuple of holdout tweets
    """
    with zipfile.ZipFile(fname) as zf:
        with io.TextIOWrapper(zf.open(path), encoding='utf-8') as f:
            X = []
            for line in f.read().split('\n')[:-1]:
                X.append(','.join(line.split(',')[1:]))
    return tuple(X)


def predict_holdout(
    clf: Any,
    out_path: str
) -> None:
    """
    predicts on the holdout set using provided classifier

    :param clf: classifier with .predict() method
    :param out_path: where to save output to
    """
    X = get_holdout()
    ids = range(1, len(X) + 1)
    predictions = clf.predict(X).to(int).tolist()
    submission = pd.DataFrame([ids, predictions], index=['Id', 'Prediction'])
    submission.T.to_csv(out_path, index=False)


def fit_and_evaluate_baseline_models(
    out_path: str,
    baseline_models: Dict[str, Dict[str, Any]],
    seed: int,
    train_size: int,
    test_size: int,
    models_mapper: Dict[str, Any]
) -> None:
    """
    trains and evaluates baseline models

    :param out_path: where to save results to
    :param baseline_models: dictionary of baseline model (name, args) tuples
    :param seed: seed for data order
    :param train_size: number of training examples
    :param test_size: number of testing examples
    :param models_mapper: mapper from model names to model classes
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    X_train, y_train, X_test, y_test = get_data(seed, train_size, test_size)
    for baseline_name, baseline_model_args in baseline_models.items():
        clf = models_mapper[baseline_name](baseline_model_args)
        clf.fit(X_train, y_train)
        res = evaluate(clf, X_train, y_train, X_test, y_test, False, None)
        res.to_csv(f'{out_path}/{baseline_name}_results-{train_size}.csv')
    
    configuration = {
        **baseline_models,
        'seed': seed,
        'train_size': train_size,
        'test_size': test_size
    }
    with open(f'{out_path}/configurations.json', 'w') as f:
        json.dump(configuration, f)


def shuffle(
    X: Tuple[str],
    y: torch.Tensor,
    seed: int
) -> Tuple[Tuple[str], torch.Tensor]:
    """
    shuffle examples based on seed

    :param X: tweets
    :param y: labels
    :param seed: seed for RNG
    :return: tuple of (X, y) shuffled
    """
    data = list(zip(X, y.tolist()))
    random.Random(seed).shuffle(data)
    X, y = zip(*data)
    y = torch.Tensor(y)
    return X, y
