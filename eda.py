import tqdm
import torch
import pandas as pd
from contextlib import redirect_stdout

from typing import Tuple, Union, Callable

from helpers import get_data
from preprocessing import drop_duplicates


def frequencies_eda(
    tweets: Tuple[str],
    labels: torch.Tensor,
    get_property: Callable,
    level: str
) -> pd.DataFrame:
    """
    computes frequencies of tweet or token properties by sentiment

    :param tweets: tweets
    :param labels: labels
    :param get_property: callable returning property or None
    :param level: either 'tweet' or 'token'
    :return: dataframe with property index, columns pos, neg, total, pos_prob
    """
    frequencies = {-1: {}, 1: {}}
    pbar = tqdm.tqdm(zip(tweets, labels.tolist()), total=len(tweets))
    for tweet, label in pbar:
        if level == 'tweet':
            prop = get_property(tweet)
            if prop is not None:
                if prop in frequencies[label]:
                    frequencies[label][prop] += 1
                else:
                    frequencies[label][prop] = 1
        elif level == 'token':
            for token in tweet.split():
                prop = get_property(token)
                if prop is not None:
                    if prop in frequencies[label]:
                        frequencies[label][prop] += 1
                    else:
                        frequencies[label][prop] = 1
        else:
            raise NotImplementedError(
                f'"level" must be "tweet" or "token", but got {level}'
            )
    idx = list(set(frequencies[-1].keys()) | set(frequencies[1].keys()))
    props = pd.DataFrame(0, index=idx, columns=[-1, 1])
    for sentiment, freq in frequencies.items():
        for prop, count in freq.items():
            props.loc[prop, sentiment] += count
    props['total'] = props.sum(axis=1)
    props['pos_prob'] = props[1] / props['total']
    props.sort_values(by='total', ascending=False, inplace=True)
    props.rename(columns={-1: 'neg', 1: 'pos'}, inplace=True)
    return props


def length_prop(
    tweet: str
) -> str:
    """
    returns tweet length rounded to 10

    :param tweet: a tweet
    :return: rounded length of tweet
    """
    bracket = len(tweet) // 10
    return f'{bracket * 10} - {bracket * 10 + 9}'


def smiley_prop(
    tweet: str,
) -> Union[str, None]:
    """
    returns smiley if believed present, else None

    :param tweet: a tweet
    :return: assumed smiley or None
    """
    falsealarm_ids = [
        '<user>', '<url>', 'live at <url>', 'via <user>', 'rt <user>'
    ]
    falsealarm_ids = [f'( {s}' for s in falsealarm_ids]

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
                return None
            else:
                return ':('
        elif '(' not in tweet and ')' in tweet:
            return ':)'
        else:
            return None
    return None


def spam_prop(
    tweet: str,
) -> Union[str, None]:
    """
    returns "spam" if tweet is a hashtag, None otherwise

    :param tweet: a tweet
    :return: hashtag or None
    """
    if '(' in tweet and tweet.endswith('... <url>'):
        return 'spam'
    return None


def hashtag_prop(
    token: str
) -> Union[str, None]:
    """
    returns token if it is a hashtag, None otherwise

    :param token: token
    :return: hashtag or None
    """
    if token.startswith('#'):
        return token
    return None


def tags_prop(
    token: str
) -> Union[str, None]:
    """
    returns tag token is a tag, None otherwise

    :param token: token
    :return: tag or None
    """
    if token == '<user>' or token == '<url>':
        return token
    return None


def onechar_prop(
    token: str
) -> Union[str, None]:
    """
    returns character if token is of length 1, None otherwise

    :param token: token
    :return: character or None
    """
    if len(token) == 1:
        return token
    return None


def conduct_frequencies_eda(
    n: int,
    min_token_frequency: float,
    data_size: int,
    out_path: str
) -> None:
    """
    runs a frequency analysis for a range of properties of tweets or tokens

    :param n: number of most common (or most biased) properties to print
    :param min_token_frequency: for minimum frequency for bias analysis
    :param data_size: on how many examples to run frequencies eda
    :param out_path: where to save results to
    """
    tweets, labels, _, _ = get_data(42, data_size, 2500000 - data_size)

    tweets_freq = frequencies_eda(tweets, labels, lambda t: t, 'tweet')
    tweets, labels = drop_duplicates(tweets, labels)
    lengths_freq = frequencies_eda(tweets, labels, length_prop, 'tweet')
    smileys_freq = frequencies_eda(tweets, labels, smiley_prop, 'tweet')
    spam_freq = frequencies_eda(tweets, labels, spam_prop, 'tweet')
    tokens_freq = frequencies_eda(tweets, labels, lambda t: t, 'token')
    common_idx = tokens_freq['total'] > len(tweets) * min_token_frequency
    tokens_freq_common = tokens_freq[common_idx]
    hashtags_freq = frequencies_eda(tweets, labels, hashtag_prop, 'token')
    tags_freq = frequencies_eda(tweets, labels, tags_prop, 'token')
    onechartokens_freq = frequencies_eda(tweets, labels, onechar_prop, 'token')

    with open(out_path, 'w') as f:
        with redirect_stdout(f):
            data_share = f'{data_size / 2500000 * 100:.0f}%'
            print(f'Frequencies Analysis on {data_share} of data:\n')
            print(f'\nTop {n} Most Frequent Tweets:')
            print(tweets_freq[:n])
            print(f'\nTop {n} Tweet Lengths:')
            print(lengths_freq[:n])
            print(f'\nTop {n} Assumed Smileys:')
            print(smileys_freq)
            print(f'\nTop {n} Assumed Spam:')
            print(spam_freq)
            print(f'\nTop {n} Most Negative Common Tokens:')
            print(tokens_freq_common.sort_values(by='pos_prob')[:n])
            print(f'\nTop {n} Most Positive Common Tokens:')
            print(tokens_freq_common.sort_values(
                by='pos_prob', ascending=False
            )[:n])
            print(f'\nTop {n} Most Frequent Hashtags:')
            print(hashtags_freq[:n])
            print('\nSpecial Tokens:')
            print(tags_freq)
            print(f'\nTop {n} Single-Character Tokens:')
            print(onechartokens_freq[:n])
