"""This is the module that actually trains the guy"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange
import tensorflow as tf

import model
import reuters

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.5, "learning rate")
flags.DEFINE_float("learning_rate_decay", 0.99, "decay lr this much")
flags.DEFINE_float("max_grad_norm", 10000.0, "clip gradients to this")
flags.DEFINE_integer("batch_size", 64, "batch size to use")
flags.DEFINE_integer("size", 256, "size of each model layer")
flags.DEFINE_integer("num_layers", 1, "number of model layers")
flags.DEFINE_integer("vocab_size", 10000, "number of words to use")
flags.DEFINE_integer("steps_per_checkpoint", 200, "how often to save")
flags.DEFINE_string("model_dir", "models", "where to save the models")

FLAGS = flags.FLAGS

# these are the bucket sizes we are going to be using.
_buckets = [10, 20, 40, 80, 160]


def create_model(session, forward_only):
    """Set up the model, initialise or load params"""
    print('...getting model...', end='')
    model = model.SequenceAutoencoder(
        FLAGS.vocab_size,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.batch_size,
        _buckets,
        FLAGS.max_grad_norm,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay,
        train=not forward_only)
    print('\r~~~~Got model.')
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print('...initialising from file...', end='')
        # initialise everything not saved
        session.run(tf.initialize_all_variables())
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print('...initialising fresh...', end='')
        session.run(tf.initialize_all_variables())
    print('\r~~~~Initialised')
    return model


def train():
    """Train a sequence autoencoder using some kind of data"""
    print('...getting data...', end='')
    # get the data
    print('\r~~~~Got data')
    
