import pandas as pd
import torch
from emoji import demojize
from nltk.tokenize import TweetTokenizer

from typing import Tuple


def replace_tags(
    tweets: Tuple[str]
) -> Tuple[str]:
    """
    adjust user and url tokens as suggested by BERTweet authors
    
    :param tweets: tweets
    :return: tweets with tags replaced
    """
    tweets_tokenshandled = []
    for tweet in tweets:
        tokens = []
        for token in tweet.split():
            if token == '<user>':
                tokens.append("@USER")
            elif token == '<url>':
                tokens.append("HTTPURL")
            else:
                tokens.append(token)
        tweets_tokenshandled.append(' '.join(tokens))
    return tuple(tweets_tokenshandled)


def drop_duplicates(
    X: Tuple[str],
    y: torch.Tensor
) -> Tuple[Tuple[str], torch.Tensor]:
    """
    removes duplicate tweets

    :param X: tweets
    :param y: labels
    :return: tuple of unique observations
    """
    X = pd.Series(X)
    y = y.numpy()
    unique_idx = ~X.duplicated()
    X = tuple(X[unique_idx].values.tolist())
    y = torch.Tensor(y[unique_idx].tolist())
    return X, y


def filter_spam(
    X: Tuple[str],
    y: torch.Tensor,
    drop: bool
) -> Tuple[Tuple[str], torch.Tensor]:
    """
    remove or extract tweets expected to be spam

    :param X: tuple of tweets
    :param y: tensor of sentiments
    :param drop: whether to remove or extract spam tweets
    :return: tuple of (X, y)
    """
    X, y = pd.Series(X), y.numpy()
    idx = X.apply(lambda t: '(' in t and t.endswith('... <url>'))
    if drop:
        idx = ~idx
    X = tuple(X[idx].values.tolist())
    y = torch.Tensor(y[idx].tolist())
    return X, y


def reconstruct_smileys(
    tweets: Tuple[str],
) -> Tuple[str]:
    """
    attempts to reconstruct smileys like ":))" or ":(("
    
    :param tweets: tweets
    :return: tweets with assumed smileys reconstructed
    """
    falsealarm_ids = [
        '<user>', '<url>', 'live at <url>', 'via <user>', 'rt <user>'
    ]
    falsealarm_ids = [f'( {s}' for s in falsealarm_ids]
    tweets_reconstructed = []
    for tweet in tweets:
        if tweet.count('(') != tweet.count(')'):
            while ') )' in tweet:
                tweet = tweet.replace(') )', ')')
            while '( (' in tweet:
                tweet = tweet.replace('( (', '(')
            if tweet.count('(') > tweet.count(')'):
                falsealarm = False
                for falsealarm_id in falsealarm_ids:
                    if falsealarm_id in tweet:
                        falsealarm = True
                        break
                if falsealarm:
                    tweets_reconstructed.append(tweet)
                else:
                    tweets_reconstructed.append(tweet.replace('(', ':('))
            elif '(' not in tweet and ')' in tweet:
                tweets_reconstructed.append(tweet.replace(')', ':)'))
            else:
                tweets_reconstructed.append(tweet)
        else:
            tweets_reconstructed.append(tweet)
    return tuple(tweets_reconstructed)


def vinai_preprocessing(
    tweets: Tuple[str]
) -> Tuple[str]:
    """
    tokenize using NLTK, translate emotion icons into text strings, normalize;
    see https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py
    
    :param tweets: raw tweets
    :return: processed tweets
    """
    replacings = [
        ("cannot ", "can not "),
        ("n't ", " n't "),
        ("n 't ", " n't "),
        ("ca n't", "can't"),
        ("ai n't", "ain't"),
        ("'m ", " 'm "),
        ("'re ", " 're "),
        ("'s ", " 's "),
        ("'ll ", " 'll "),
        ("'d ", " 'd "),
        ("'ve ", " 've "),
        (" p . m .", "  p.m."),
        (" p . m ", " p.m "),
        (" a . m .", " a.m."),
        (" a . m ", " a.m ")
    ]
    tokenizer = TweetTokenizer()
    normalized_tweets = []
    for tweet in tweets:
        tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
        normalized_tokens = []
        for token in tokens:
            if token == '<user>':
                normalized_token = "@USER"
            elif token == '<url>':
                normalized_token = "HTTPURL"
            elif len(token) == 1:
                normalized_token = demojize(token)
            else:
                if token == "’":
                    normalized_token = "'"
                elif token == "…":
                    normalized_token = "..."
                else:
                    normalized_token = token
            normalized_tokens.append(normalized_token)
        normalized_tweet = ' '.join(normalized_tokens)
        for replacee, replacer in replacings:
            normalized_tweet = normalized_tweet.replace(replacee, replacer)
        normalized_tweet = ' '.join(normalized_tweet.split())
        normalized_tweets.append(normalized_tweet)
    return tuple(normalized_tweets)


def preprocess(
    X: Tuple[str],
    y: torch.Tensor
) -> Tuple[Tuple[str], torch.Tensor]:
    """
    perform preprocessing

    :param X: tweets
    :param y: labels
    :return: tuple of (X, y)
    """
    X, y = drop_duplicates(X, y)
    X, y = filter_spam(X, y, drop=True)
    X = vinai_preprocessing(X)
    return X, y
