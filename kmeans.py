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
    Assumes a and b are rank 1 tensors or a batch of them"""
    with tf.name_scope(name):
        prod = tf.matmul(a, b, transpose_b=True)
        a_norm = l2_norm(a)
        b_norm = l2_norm(b)
        return prod / (a_norm * b_norm)


def euclidean(a, b, name='euclidean-dist'):
    """Finds the euclidean distance between a and b ie the l2 norm of the
    difference. Returns pair-wise"""
    with tf.name_scope(name):
        num_bs = b.get_shape()[0].value
        num_as = a.get_shape()[0].value
        dims = a.get_shape()[1].value  # assume the same for a and b
        tile_a = tf.reshape(tf.tile(a, [1, num_bs]), [num_as, num_bs, dims])
        tile_b = tf.reshape(tf.tile(b, [num_as, 1]), [num_as, num_bs, dims])
        return l2_norm(tile_a - tile_b, reduction_indices=2)


def cluster(data, num_means, max_iters=1000, distance=cosine_similarity):
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
    dists = distance(data, centroids)
    # we now have a row for each data point
    # each row has k values which are the distances
    new_assignments = tf.argmin(dists, 1)
    assignment_change = tf.reduce_any(tf.not_equal(assignments,
                                                   new_assignments))
    # now we have to calculate new means
    sums = tf.unsorted_segment_sum(data, new_assignments, num_means)
    counts = tf.unsorted_segment_sum(tf.ones_like(data), new_assignments, num_means)
    new_means = sums / counts
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
    data = np.random.multivariate_normal(np.array([-5, -5]),
                                         cov=np.identity(2),
                                         size=(500))
    data = np.vstack(
        (data, np.random.multivariate_normal(np.zeros(2)+10,
                                             cov=np.identity(2),
                                             size=(300))))
    
    means, clusters = cluster(data, 2, distance=euclidean)
    print(means)
