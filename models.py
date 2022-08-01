import tqdm
import torch
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from simpletransformers.classification import ClassificationModel
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier

from typing import Dict, Tuple, Any


class BertweetClassifier:
    """
    finetuning BERTweet-base via simpletransformers
    """
    def __init__(
        self,
        model_args: Dict[str, Any]
    ) -> None:
        """
        initialization

        :param model_args: model hyperparameters
        """
        self.model = ClassificationModel(
            model_type='bertweet',
            model_name='vinai/bertweet-base',
            args=model_args,
            num_labels=2,
            use_cuda=True
        )

    def fit(
        self,
        X: Tuple[str],
        y: torch.Tensor
    ) -> None:
        """
        finetune BERTweet

        :param X: tuple of tweets
        :param y: tensor of sentiments
        """
        train_df = pd.DataFrame({'text': X, 'labels': (y + 1) / 2})
        self.model.train_model(train_df=train_df, acc=accuracy_score)
    
    def predict_proba(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        """
        predict probabilities

        :param X: tuple of tweets
        """
        return torch.Tensor(softmax(self.model.predict(X)[1], axis=1))

    def predict(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        """
        predict sentiments

        :param X: tuple of tweets
        """
        return 2 * self.predict_proba(X)[:, 1].round() - 1


class TokenTupleFrequenciesClassifier:
    def __init__(
        self,
        model_args: Dict[str, Any]
    ) -> None:
        """
        initializes classifier

        :param model_args:
            n: the length of the word tuple
            p: the power by which the frequency is weighted within the score
        """
        self.n = model_args['n']
        self.p = model_args['p']
        self.n_tuple_counters = {-1: dict(), 1: dict()}
        self.frequencies = {-1: dict(), 1: dict()}

    def fit(
        self,
        X: Tuple[str],
        y: torch.Tensor
    ) -> None:
        """
        computes an occurrence counter for each consecutive n-tuple of words
        for each sentiment, creates a score by scaling the count by power p
        
        :param X: list of tweets
        :param y: tensor of sentiments
        """
        pabr = tqdm.tqdm(zip(X, y), total=len(X), desc='Computing Frequencies')
        for tweet, label in pabr:
            words, sentiment = tweet.split(), label.item()
            if len(words) >= self.n:
                words_offset = []
                for i in range(self.n):
                    words_offset.append(words[i:len(words) - self.n + i + 1])
                for n_tuple_tuple in zip(*words_offset):
                    n_tuple = ' '.join(n_tuple_tuple)
                    if n_tuple in self.n_tuple_counters[sentiment]:
                        self.n_tuple_counters[sentiment][n_tuple] += 1
                    else:
                        self.n_tuple_counters[sentiment][n_tuple] = 1
        for sentiment, counter in self.n_tuple_counters.items():
            for n_tuple, count in counter.items():
                self.frequencies[sentiment][n_tuple] = count ** self.p

    def predict_proba(
        self,
        X: Tuple[str],
    ) -> torch.Tensor:
        """
        creates a score for each tweet and each sentiment as the sum of scores
        for each consecutive n-tuple of words in the tweet of that sentiment,
        calculates the probability for positive label as the share of positive
        score to overall score (or as 0.5, if overall score is 0)

        :param X: list of tweets
        :return: a tensor of predicted probabilities for positive label
        """
        probabilities = []
        pbar = tqdm.tqdm(X, total=len(X), desc='Computing Scores')
        for tweet in pbar:
            scores = {-1: 0, 1: 0}
            words = tweet.split()
            if len(words) >= self.n:
                words_offset = []
                for i in range(self.n):
                    words_offset.append(words[i:len(words) - self.n + i + 1])
                for n_tuple_tuple in zip(*words_offset):
                    n_tuple = ' '.join(n_tuple_tuple)
                    for sentiment in [-1, 1]:
                        if n_tuple in self.frequencies[sentiment]:
                            score = self.frequencies[sentiment][n_tuple]
                            scores[sentiment] += score
            if scores[-1] == 0 and scores[1] == 0:
                prob = 0.5
            else:
                prob = scores[1] / (scores[-1] + scores[1])
            probabilities.append(prob)
        probabilities = torch.Tensor(probabilities)
        probabilities = torch.stack((1 - probabilities, probabilities), 1)
        return probabilities

    def predict(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        """
        predicts labels

        :param X: List of tweets as strings
        :return: a tensor of predicted probabilities for positive label
        """
        return 2 * self.predict_proba(X)[:, 1].round() - 1


class EmbeddingsBoostingClassifier:
    def __init__(
        self,
        model_args: Dict
    ) -> None:
        """
        initialize model

        :param model_args: model arguments
            val_size: number of examples in validation set
            verbose: verbosity
            early_stopping_rounds: number of early stopping rounds
            transformer_name: name of model which to use to create embeddings

        """
        self.val_size = model_args['val_size']
        self.verbose = model_args['verbose']
        self.embedder = SentenceTransformer(model_args['transformer_name'])
        self.model = CatBoostClassifier(
            early_stopping_rounds=model_args['early_stopping_rounds'],
            task_type='GPU',
            devices='0:1'
        )

    def fit(
        self,
        X: Tuple[str],
        y: torch.Tensor
    ) -> None:
        """
        train CatBoost Classifier

        :param X: list of tweets
        :param y: tensor of sentiments
        """
        X_embedded = self.embedder.encode(X, show_progress_bar=True)
        y_ = y.numpy()
        split_idx = int(len(X) * (1 - self.val_size))
        X_train, y_train = X_embedded[:split_idx], y_[:split_idx]
        X_val, y_val = X_embedded[split_idx:], y_[split_idx:]
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=self.verbose
        )

    def predict_proba(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        """
        predict probabilities

        :param X: list of tweets
        :return: tensor of probabilities for positive label
        """
        X_embedded = self.embedder.encode(X, show_progress_bar=True)
        probabilities = self.model.predict_proba(X=X_embedded)
        return torch.Tensor(probabilities)

    def predict(
        self,
        X: Tuple[str]
    ) -> torch.Tensor:
        """
        predict labels

        :param X: list of tweets
        :return: tensor of sentiments
        """
        return 2 * self.predict_proba(X)[:, 1].round() - 1


MODELS_MAPPER = {
    'BertweetClassifier': BertweetClassifier,
    'TokenTupleFrequenciesClassifier': TokenTupleFrequenciesClassifier,
    'EmbeddingsBoostingClassifier': EmbeddingsBoostingClassifier
}
