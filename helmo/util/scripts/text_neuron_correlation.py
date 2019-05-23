import os
import pickle
import argparse
from collections import Counter
import gc

import numpy as np


def mark_up_spaces_with_pos(words, tags, text_len):
    markup = [0] * text_len
    idx = 0
    for i in range(len(words)-1):
        idx += len(words[i])
        try:
            markup[idx] = tags[i]
        except IndexError:
            print(i, idx)
            raise
        idx += 1
    return markup


def mark_up_spaces_with_pos_in_text8(words, tags, text_len):
    return [0] + mark_up_spaces_with_pos(words, tags, text_len-1)


def get_markup_for_1_tag(markup, tag):
    markup = markup.copy()
    for i, t in enumerate(markup):
        if t != 0:
            markup[i] = 1 if t == tag else -1
    return markup


# def get_stats_by_tags(data, markup, tags):
#     stats = {}
#     for tag in tags:
#         markup_for_tag = get_markup_for_1_tag(markup, tag)
#         stats[tag] = compute_stats(data, markup_for_tag)
#     return stats


def get_relevant_activations(data, markup_for_tag):
    result = []
    for i, tag in enumerate(markup_for_tag):
        if tag != 0:
            result.append(data[i])
    return np.stack(result)


def get_matches(activations, markup):
    markup = np.array(markup)
    markup_devs = markup - np.mean(markup)
    activation_devs = activations - np.mean(activations, 0, keepdims=True)
    activation_stddevs = np.std(activations, 0, ddof=1, keepdims=True)
    markup_stddev = np.std(markup, ddof=1)
    activation_dev_fractions = activation_devs / (activation_stddevs + 1e-20)
    markup_dev_fractions = markup_devs / (markup_stddev + 1e-20)
    return activation_dev_fractions * np.reshape(markup_dev_fractions, [-1, 1])


def compute_stats(data, markup_for_tag):
    stats = {}
    stats['markup'] = markup_for_tag
    stats['relevant_markup'] = list(filter(lambda x: x != 0, markup_for_tag))
    stats['relevant_activations'] = get_relevant_activations(data, markup_for_tag)
    stats['matches'] = get_matches(stats['relevant_activations'], stats['relevant_markup'])
    stats['correlations'] = np.mean(stats['matches'], 0)
    assert stats['correlations'].ndim == 1
    stats['match_stddevs'] = np.std(stats['matches'], 0)
    stats['mean_square_correlation'] = np.sqrt(np.mean(stats['correlations']**2))
    stats['meta'] = {
        "positive": np.count_nonzero(markup_for_tag == 1),
        "negative": np.count_nonzero(markup_for_tag == -1),
        "total": len(stats['markup']),
    }
    return stats


def save_tag_stats(stats, directory):
    os.makedirs(directory, exist_ok=True)
    for stat, value in stats.items():
        path = os.path.join(directory, stat + '.pickle')
        with open(path, 'wb') as f:
            pickle.dump(value, f)


# def save_stats_by_tags(stats, directory):
#     os.makedirs(directory, exist_ok=True)
#     for tag, tag_stats in stats.items():
#         save_tag_stats(tag_stats, os.path.join(directory, tag))


def replace_None_tag_with_zero(words_with_tags):
    for i, wt in enumerate(words_with_tags):
        if wt[1] is None:
            words_with_tags[i] = (wt[0], 0)


def get_and_save_stats_for_tag(data, markup, tag, directory):
    markup_for_tag = get_markup_for_1_tag(markup, tag)
    stats = compute_stats(data, markup_for_tag)
    save_tag_stats(stats, directory)


def get_and_save_stats(data, markup, tags, directory):
    os.makedirs(directory, exist_ok=True)
    for tag in tags:
        print("Processing tag {}".format(tag))
        get_and_save_stats_for_tag(data, markup, tag, os.path.join(directory, tag))


def remove_tags_with_small_counts(counter):
    for tag in counter:
        if counter[tag] < 2:
            del counter[tag]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="The script is for measuring correlation between neurons activation "
                    "and text features. Inputs are a text on which model is tested and "
                    "array of neuron activations measured on each character of input text. "
                    ""
                    "The script finds matches between specified features and neuron "
                    "activations and saves them in `matches.pickle`. Matches are products "
                    "of devs of activations from their means divided by stddevs and devs "
                    "of markup from markup means divided by stddevs of markup."
                    ""
                    "In addition, correlations between features and activations are "
                    "computed and saved to `correlations.pickle`. "
                    "Correlations are mean values of matches. "
                    ""
                    "In order to understand how far from zero are these correlations, "
                    "matches stddevs are computed and saved into `match_stddevs.pickle`. "
                    "Match is divided by stddevs of activation and feature before stddev "
                    "computation. "
                    ""
                    "Mean square of feature correlation for a layer is computed. This "
                    "are saved in `mean_square_correlation.pickle`. "
                    ""
                    "Of course initial feature maps are also can be saved as `markup.pickle`. "
                    ""
                    "File `meta.txt` contains total number of characters `\"total\"`, "
                    "number of positive marks `\"positive\"`, number of negative marks "
                    "`\"negative\"`."
                    ""
                    "All listed files are put in directories named after recognised features."
    )
    parser.add_argument(
        "text_file",
        help="File with text from which features are taken."
    )
    parser.add_argument(
        "--start",
        help="Index of the first character in analyzed substring of text. Default is zero.",
        default=0,
    )
    parser.add_argument(
        "--length",
        help="Length of analyzed substring of text. Default is 6.4e5.",
        default=640000,
    )
    parser.add_argument(
        "markup",
        help="The way text is marked up. Possible options are: (1)"
             "`space_pos`."
    )
    parser.add_argument(
        "--data",
        help="Pickle file with one 2D numpy array."
    )
    parser.add_argument(
        "--tags",
        "-t",
        help="Tags to process. You should specify at list one.",
        nargs='+',
    )
    parser.add_argument(
        "--tagged_words_file",
        "-w",
        help="File with tagged words.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to directory with results",
        default=".",
    )
    args = parser.parse_args()
    with open(args.text_file, 'rb') as f:
        text = f.read()
    with open(args.tagged_words_file, 'rb') as f:
        words_with_tags = pickle.load(f)
    replace_None_tag_with_zero(words_with_tags)
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    words, tags = zip(*words_with_tags)

    markup = mark_up_spaces_with_pos_in_text8(words, tags, len(text))
    markup = markup[args.start:args.start + args.length]
    tag_counter = Counter(tags)
    del tag_counter[0]
    remove_tags_with_small_counts(tag_counter)
    unique_tags = list(tag_counter.keys()) if args.tags is None else args.tags
    del words_with_tags, words
    # gc.collect()
    get_and_save_stats(data, markup, unique_tags, args.output)
    # stats_by_tags = get_stats_by_tags(data, markup, tags)
    # save_stats_by_tags(stats_by_tags, args.output)
