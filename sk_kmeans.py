"""Use scikit for clustering :)"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from collections import Counter
import logging
import os

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

import reuters
import nn_encode


def get_args():
    parser = argparse.ArgumentParser(description='Do some clustering')
    parser.add_argument('--num_clusters', type=int, help='how many clusters',
                        default=0)
    parser.add_argument('--num_features', type=int,
                        help='how big the tf idf vectors are',
                        default=20000)
    parser.add_argument('--nn_features', type=bool,
                        help='use neural net or tf-idf',
                        default=False)
    parser.add_argument('--max_len', type=int,
                        help='maximum length, for fairness',
                        default=200)
    parser.add_argument('--nn_vecfile', type=str,
                        help='if nn_features, then where to find them. If '
                             'not found, will be generated and stored here',
                        default='nn_features')
    parser.add_argument('--compare', type=bool,
                        help='if true, then do both clusterings and compare them',
                        default=False)
    return parser.parse_args()


def get_nn_filenames(prefix):
    """Gets the filenames for the nn features"""
    return prefix+'-train.np.gz', prefix+'-test.np.gz'


def filter_length(data_a, data_b, maxlen):
    """Filters a pair of lists based on length of items in the first"""
    b_transform = [b for a, b in zip(data_a, data_b)
                   if len(a.split()) < maxlen]
    a_transform = [a for a in data_a if len(a.split()) < maxlen]
    return a_transform, b_transform


def normalise(mat):
    """Normalise a matrix (l2) along its rows"""
    return mat / (np.sum(mat**2, axis=1, keepdims=True))


def get_clustering(data, num_clusters):
    """Does a clustering, returns the object"""
    clusterer = KMeans(n_clusters=num_clusters)
    clusterer.fit(data)
    return clusterer


def get_tf_idf(train, test, max_features):
    """Transforms data to tf-idf"""
    print('~~Transforming to tf-idf')
    vectorizer = TfidfVectorizer(max_df=0.5,
                                 max_features=max_features,
                                 min_df=2,
                                 stop_words='english')
    start = time.time()
    X = vectorizer.fit_transform(train)
    tX = vectorizer.transform(test)
    print('~~~~(done in {}s)'.format(time.time() - start))
    print('~~~~(result is {})'.format(X.shape))
    return X, tX


def get_nn_features(train, test, file_prefix):
    """Transforms data using nn or
    loads from file"""
    train_filename, test_filename = get_nn_filenames(file_prefix)
    if not os.path.exists(train_filename):
        print('~~Transforming via nn')
        start = time.time()
        X, tX = [], []
        for i, words in enumerate(train):
            X.append(nn_encode.get_vector(words))
            if i % 100 == 0:
                print('\r~~~~train: {}/{}'.format(i+1, len(train)), end='')
        for i, words in enumerate(test):
            tX.append(nn_encode.get_vector(words))
            if i % 100 == 0:
                print('\r~~~~test : {}/{}'.format(i+1, len(test)), end='')
        print()
        X = np.array(X)
        tX = np.array(tX)
        print('~~~~(done in {}s)'.format(time.time() - start))
        print('~~~~(result is {})'.format(X.shape))
        np.savetxt(train_filename, X)
        np.savetxt(test_filename, tX)
        print('~~~~(saved results for later)')
    else:
        print('~~Loading from {}'.format(train_filename))
        start = time.time()
        X = np.loadtxt(train_filename)
        tX = np.loadtxt(test_filename)
        print('~~~~(done in {}s)'.format(time.time() - start))
    return X, tX


def solo_evaluate(clusterer, X, labels, tX, tlabels):
    """Prints some info about a given clustering"""
    print('~~~Silhouette score: {}'.format(
        metrics.silhouette_score(X, clusterer.labels_, sample_size=1000)))
    print()
    print('~~Top 5 labels per cluster:')
    assignment = clusterer.labels_
    counts_per_cluster = Counter(assignment)
    label_counts = [Counter() for _ in clusterer.cluster_centers_[:, ...]]
    for i, labelling in enumerate(labels):
        for label in labelling:
            label_counts[assignment[i]][label] += 1
    for i, count in enumerate(label_counts):
        print('~~~{} ({}): '.format(i, counts_per_cluster[i]), end='')
        print(', '.join(['{} ({})'.format(item[0], item[1])
                         for item in count.most_common(5)]))
    # try accuracy as classifier
    guesses = clusterer.predict(tX)
    # how do we get the accuracy?
    # let's go with something like:
    #    percentage of labels it guess on average
    #    taking a prediction to be the 10 most common labels
    #    in the cluster
    accumulator = 0
    cluster_labels = [[label for label, count in counter.most_common(5)]
                      for counter in label_counts]
    for i, guess in enumerate(guesses):
        # how many of the labels did it get correct?
        ground_truth = tlabels[i]
        # print(cluster_labels[guess], ground_truth)
        if len(ground_truth) == 0:
            continue  # some of them have no labels
        # count the labels in both
        hits = len([label
                    for label in ground_truth
                    if label in cluster_labels[guess]])
        accumulator += hits / len(ground_truth)
    print('~~Accuracy on new data:')
    print('~~~~average chance of correct labels (top 5): {}'.format(
        accumulator / len(tlabels)))


def compare_clusters(nn_clusters, tf_clusters):
    """prints some comparisons"""
    print('~~Adjusted mutual information: {}'.format(
        metrics.adjusted_mutual_info_score(nn_clusters.labels_,
                                           tf_clusters.labels_)))
    print('~~Normalized mutual information: {}'.format(
        metrics.normalized_mutual_info_score(nn_clusters.labels_,
                                             tf_clusters.labels_)))
    print('~~Adjusted Rand Index: {}'.format(
        metrics.adjusted_rand_score(nn_clusters.labels_,
                                    tf_clusters.labels_)))


def main():
    args = get_args()
    # load the data
    print('~~Getting data')
    data, labels = reuters.get_text_and_labels(split='train')
    tdata, tlabels = reuters.get_text_and_labels(split='test')
    # cut it down to appropriate size
    data, labels = filter_length(data, labels, args.max_len)
    tdata, tlabels = filter_length(tdata, tlabels, args.max_len)

    unique_labels = set()
    for lab in labels:
        unique_labels.update(lab)
    num_labels = len(unique_labels)
    print('~~~~({} labels in total)'.format(num_labels))

    if not args.compare:  # just do one
        if not args.nn_features:
            X, tX = get_tf_idf(data, tdata, args.num_features)
        else:
            X, tX = get_nn_features(data, tdata, args.nn_vecfile)
        print('~~Starting clustering')
        start = time.time()
        clusterer = get_clustering(X, args.num_clusters or num_labels)
        print('~~~~(done in {}s)'.format(time.time() - start))

        # try and evaluate the clusters
        solo_evaluate(clusterer, X, labels, tX, tlabels)
    else:
        nnX, nntX = get_nn_features(data, tdata, args.nn_vecfile)
        tfX, tftX = get_tf_idf(data, tdata, args.num_features)
        print('~~Clustering NN')
        start = time.time()
        nn_clusterer = get_clustering(nnX, args.num_clusters or num_labels)
        print('~~~~(done in {}s)'.format(time.time() - start))
        start = time.time()
        tf_clusterer = get_clustering(tfX, args.num_clusters or num_labels)
        print('~~~~(done in {}s)'.format(time.time() - start))
        print('/////////////////////////////////////////////')
        print('\\\\\\\\\\\\Evaluation of NN clusters alone:')
        solo_evaluate(nn_clusterer, nnX, labels, nntX, tlabels)
        print('/////////////////////////////////////////////')
        print('\\\\\\\\\\\\Evaluation of tf-idf clusters alone:')
        solo_evaluate(tf_clusterer, tfX, labels, tftX, tlabels)
        print('//////////////////////////////////////////////')
        print('\\\\\\\\\\\\\\\\\\\\\\Both')
        compare_clusters(nn_clusterer, tf_clusterer)


if __name__ == '__main__':
    # logging.getLogger().setLevel(logging.INFO)
    main()
