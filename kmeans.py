"""Tensorflow k-means implementation."""
import time

import numpy as np
import tensorflow as tf


def l2_norm(a, reduction_indices=None, name='l2-norm'):
    """Returns l2 norm of a tensor"""
    with tf.name_scope(name):
        return tf.sqrt(tf.reduce_sum(a*a, reduction_indices))


def cosine_similarity(a, b, name='cos-sim'):
    """Construct ops that return the cosine distance between a and b.
    Assumes a and b are rank 1 tensors."""
    with tf.name_scope(name):
        prod = tf.matmul(a, b, transpose_b=True)
        a_norm = l2_norm(a)
        b_norm = l2_norm(b)
        return prod / (a_norm * b_norm)


def cluster(data, num_means, max_iters=1000):
    """Clusters the data using k-means.

    Args:
        data: the data, either one big numpy ndarray of shape
            `(data_size, data_length)` or a list of arrays of size
            `(data_length)`.
        num_means: the number of clusters to attempt to fit.
        max_iters: the maximum number of iterations to carry out.

    Returns:
        (list of means, list of clusters). Each cluster is a list of ndarray.
    """
    start = time.time()
    data_size, num_features = data.shape
    np_data = data
    data = tf.Variable(np.array(data))
    assignments = tf.Variable(tf.zeros([data_size], dtype=tf.int64))

    # initialise the centroids in some way
    # should do better
    centroids = tf.Variable(tf.slice(data.initialized_value(),
                                     [0, 0],
                                     [num_means, num_features]))
    # next we need to figureout how to compute the similarity between each
    # centroid and each data point
    # cos similarity actually makes this pretty handy
    # we want the dot product of each row of data with each row of
    # centroids.
    dots = tf.matmul(tf.nn.l2_normalize(data, 0),
                     tf.nn.l2_normalize(centroids, 0),
                     transpose_b=True)
    # we now have a row for each data point
    # each row has k values which are the dot products.
    # because we normalized first, they are genuinely the cosine similarities.
    # so we take the arg max across columns to get the new assignments
    new_assignments = tf.argmax(dots, 1)
    assignment_change = tf.reduce_any(tf.not_equal(assignments,
                                                   new_assignments))
    # now we have to calculate new means
    sums = tf.unsorted_segment_sum(data, new_assignments, num_means)
    new_means = sums / tf.to_double(tf.shape(sums)[0])
    # all that is left is setting up some ops to ensure that we update
    # variables.
    with tf.control_dependencies([assignment_change]):
        update = tf.tuple([centroids.assign(new_means),
                           assignments.assign(new_assignments)])

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    # good to go
    init_end = time.time()

    for i in range(max_iters):
        print('\r..working ({}/{})'.format(i+1, max_iters), end='')
        changed, _, _ = sess.run([assignment_change, *update])
        if not changed:
            break
    print()

    # we need to return these in some form
    centers, assignments = sess.run([centroids, assignments])
    clusters = [[] for _ in range(num_means)]
    for i, assg in enumerate(assignments):
        clusters[assg].append(np_data[i])
    print('All done.')
    print('({}s setup)'.format(init_end - start))
    print('({}s fitting)'.format(time.time() - init_end))
    return centers, clusters


if __name__ == '__main__':
    data = np.random.uniform(high=100.0, size=(5000, 200))
    means, clusters = cluster(data, 60)
