"""Use scikit for clustering :)"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

import reuters


def get_args():
    parser = argparse.ArgumentParser(description='Do some clustering')
    parser.add_argument('--num_clusters', type=int, help='how many clusters',
                        default=8)
    parser.add_argument('--num_features', type=int, help='how big the tf idf vectors are',
                        default=20000)
    return parser.parse_args()


def main():
    args = get_args()
    clusterer = KMeans(n_clusters=args.num_clusters)
    # load the data
    print('~~Getting data')
    data, labels = reuters.get_text_and_labels()
    print('~~Transforming to tf-idf')
    vectorizer = TfidfVectorizer(max_df=0.5,
                                 max_features=args.num_features,
                                 min_df=2,
                                 stop_words='english')
    start = time.time()
    X = vectorizer.fit_transform(data)
    print('~~~~(done in {}s)'.format(time.time() - start))
    print('~~~~(result is {})'.format(X.shape))
    print('~~Starting clustering')
    start = time.time()
    clusterer.fit(X)
    print('~~~~(done in {}s)'.format(time.time() - start))
    # report some stats maybe
    
    print('~~~Silhouette?: {}'.format(
        metrics.silhouette_score(X, clusterer.labels_, sample_size=1000)))
    print()
    print('~~Top terms per cluster:')
    terms = vectorizer.get_feature_names()
    centers = clusterer.cluster_centers_.argsort()[:, ::-1]
    for i in range(args.num_clusters):
        print('~~~{}: '.format(i), end='')
        for ind in centers[i, :10]:
            print('{} '.format(terms[ind]), end='')
        print()


if __name__ == '__main__':
    main()
