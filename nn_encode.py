"""Load up the encoder half of the neural net model and run it through
Reuters, writing out the states."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import numpy as np
import tensorflow as tf

import reuters
import model as sa
import interact

flags = tf.app.flags
flags.DEFINE_string('model_dir', 'models', 'where the models live')
flags.DEFINE_string('model_file', '', 'a specific file you would prefer.')
FLAGS = flags.FLAGS


def load_encoder(vocab, buckets, size=256, num_layers=2):
    """Loads just the parts of the model we need to be able to get at
    the final state of the encoder net"""
    model = sa.SequenceEncoder(len(vocab), size, num_layers, buckets)
    return model


def initialise(session, model_dir, model_file=''):
    """Tries to initialise from a file. Might fail, in which case we are in a
    bit of trouble."""
    saver = tf.train.Saver()  # hopefully everything is in here
    if not model_file:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.gfile.Exists(ckp.model_checkpoint_path):
            model_file = ckpt.model_checkpoint_path
        else:
            raise ValueError('No model file and could not find.')
    print('~~initialising form {}'.format(model_file), end='', flush=True)
    saver.restore(session, model_file)
    print('\r~~initialised~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def main(_):
    model = load_encoder(FLAGS.model_dir, FLAGS.model_file)
    sess = tf.Session()
    with sess.as_default():
        initialise(sess, FLAGS.model_dir, FLAGS.model_file)
        print('this is where we would load the data etc')

if __name__ == '__main__':
    tf.app.run()
