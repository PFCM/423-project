"""This is the module that actually trains the guy"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange
import tensorflow as tf

import model as sa
import reuters

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.1, "learning rate")
flags.DEFINE_float("learning_rate_decay", 0.99, "decay lr this much")
flags.DEFINE_float("max_grad_norm", 2.0, "clip gradients to this")
flags.DEFINE_integer("batch_size", 64, "batch size to use")
flags.DEFINE_integer("size", 256, "size of each model layer")
flags.DEFINE_integer("num_layers", 2, "number of model layers")
flags.DEFINE_integer("vocab_size", 20000, "number of words to use")
flags.DEFINE_integer("steps_per_checkpoint", 200, "how often to save")
flags.DEFINE_string("model_dir", "models", "where to save the models")
flags.DEFINE_integer("max_steps", 100000, "how many times to run through the data")

FLAGS = flags.FLAGS

# these are the bucket sizes we are going to be using.
_buckets = [10, 20, 40, 80, 160, 320]


def bucket_data(data_seqs):
    """Puts the data into buckets. Ignores anything that doesn't fit
    (ie. length > _buckets[-1]).

    Args:
        data_seqs: list of lists of ints, the data sequences.

    Returns:
        dataset: a list of length len(_buckets); dataset[n] has a list
        of (source, target) pairs, although they are in fact the same thing.
    """
    dataset = [[] for _ in _buckets]
    for seq in data_seqs:
        # choose a bucket
        for bucket_id, bucket_size in enumerate(_buckets):
            if len(seq) < bucket_size:
                dataset[bucket_id].append([seq, seq])
                break
            # if we don't find a bucket that fits, we fall out of the inner
            # loop, successfully ignoring it.
    return dataset


def create_model(session, forward_only, vocab_size):
    """Set up the model, initialise or load params"""
    print('...getting model...', end='', flush=True)
    model = sa.SequenceAutoencoder(
        vocab_size,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.batch_size,
        _buckets,
        FLAGS.max_grad_norm,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay,
        train=not forward_only)
    print('\r                 \r~~~~\n~~~~Got model.')
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print('...initialising from file...', end='')
        # initialise everything not saved
        session.run(tf.initialize_all_variables())
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('...initialising fresh...', end='')
        session.run(tf.initialize_all_variables())
    print('\r                             \r~~~~\n~~~~Initialised')
    return model


def train():
    """Train a sequence autoencoder using some kind of data"""
    print('...getting data...', end='', flush=True)
    # get the data
    train, test, vocab = reuters.get_reuters(most_common=FLAGS.vocab_size)
    print('\r                  \r~~~~\n~~~~Got data')

    with tf.Session() as sess:
        # get model
        model = create_model(sess, False, len(vocab))
        # bucket it up
        print('...bucketing data...', end='', flush=True)
        # for now, just a few for testing
        bucketed_data = bucket_data([item[0] for item in train])
        train_bucket_sizes = [len(bucketed_data[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        print('\r~~~~Organised the data ({:.0f} records)'.format(train_total_size))

        # I'm not quite sure how this works right now
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        # get ready for actual training
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        print('...step {:>9}'.format(0), end='', flush=True)
        for step in xrange(FLAGS.max_steps):
            # choose a bucket
            random_num = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_num])
            # get a batch
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                bucketed_data, bucket_id, reuters.get_special_ids())
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            if np.isnan(loss):
                print('Aborting -- loss is nan')
                raise ValueError()

            print('\r...step {:>9}(loss: {:.3f})        '.format(
                step, loss/current_step),
                  end='', flush=True)

            # periodically save
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print()
                print('/'*80)
                print('\\\\\\\\\\\\global step: {}'.format(model.global_step.eval()))
                print('////learning rate: {:.5f}'.format(model.learning_rate.eval()))
                print('\\\\\\\\\\\\\\\\step time: {:.1f}'.format(step_time))
                print('///////perplexity: {:.5f}'.format(perplexity))
                # decrease learning rate if necessary
                if len(previous_losses) > 2 and loss > max(previous_loss[-3:]):
                    sess.run(model.learning_rate_decay_op)
                    print('/dropped learning rate/')
                save_path = os.path.join(FLAGS.model_dir, 'sequence_autoencoder.ckpt')
                model.saver.save(sess, save_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                print('\\'*80)

def main(_):
    logging.getLogger().setLevel(logging.INFO)
    train()


if __name__ == '__main__':
    tf.app.run()
