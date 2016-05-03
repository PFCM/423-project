"""Use scikit for clustering :)"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

import reuters
import nn_encode


def get_args():
    parser = argparse.ArgumentParser(description='Do some clustering')
    parser.add_argument('--num_clusters', type=int, help='how many clusters',
                        default=8)
    parser.add_argument('--num_features', type=int, help='how big the tf idf vectors are',
                        default=20000)
    parser.add_argument('--nn_features', type=bool, help='use neural net or tf-idf',
                        default=False)
    parser.add_argument('--max_len', type=int, help='maximum length, for fairness',
                        default=200)
    return parser.parse_args()


def get_nn_features():
    """Gets features from the learned neural model"""
    pass


def main():
    args = get_args()
    clusterer = KMeans(n_clusters=args.num_clusters)
    # load the data
    print('~~Getting data')
    data, labels = reuters.get_text_and_labels()
    # cut it down to appropriate size
    labels = [item for i, item in enumerate(labels)
              if len(data[i].split()) < args.max_len]
    data = [item for item in data
            if len(item.split()) < args.max_len]
    if not args.nn_features:
        print('~~Transforming to tf-idf')
        vectorizer = TfidfVectorizer(max_df=0.5,
                                     max_features=args.num_features,
                                     min_df=2,
                                     stop_words='english')
        start = time.time()
        X = vectorizer.fit_transform(data)
        print('~~~~(done in {}s)'.format(time.time() - start))
        print('~~~~(result is {})'.format(X.shape))
    else:
        print('~~Transforming via nn')
        start = time.time()
        X = []
        for i, words in enumerate(data):
            X.append(nn_encode.get_vector(words))
            if i % 100 == 0:
                print('\r~~~~{}/{}'.format(i+1, len(data)), end='')
        print()
        X = np.array(X)
        print('~~~~(done in {}s)'.format(time.time() - start))
        print('~~~~(result is {})'.format(X.shape))
    print('~~Starting clustering')
    start = time.time()
    clusterer.fit(X)
    print('~~~~(done in {}s)'.format(time.time() - start))

    # try and evaluate the clusters
    
    print('~~~Silhouette?: {}'.format(
        metrics.silhouette_score(X, clusterer.labels_, sample_size=1000)))
    print()
    print('~~Top labels per cluster:')
    assignment = clusterer.labels_
    label_counts = [Counter() for _ in clusterer.cluster_centers_[:,...]]
    for i, labelling in enumerate(labels):
        for label in labelling:
            label_counts[assignment[i]][label] += 1
    for i, count in enumerate(label_counts):
        print('~~~{}: '.format(i), end='')
        print(', '.join(['{} ({})'.format(item[0].decode(), item[1])
                         for item in count.most_common(10)]))
    # terms = vectorizer.get_feature_names()
    # centers = clusterer.cluster_centers_.argsort()[:, ::-1]
    # for i in range(args.num_clusters):
    #     print('~~~{}: '.format(i), end='')
    #     for ind in centers[i, :10]:
    #         print('{} '.format(terms[ind]), end='')
    #     print()


if __name__ == '__main__':
    main()
