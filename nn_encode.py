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

flags = tf.app.flags
flags.DEFINE_string('model_dir', 'bigvocab_untied_models',
                    'where the models live')
flags.DEFINE_string('model_file', '', 'a specific file you would prefer.')
flags.DEFINE_integer('vocab_size', 20000, 'size of vocab')
FLAGS = flags.FLAGS

_model = None
_sess = None
_vocab = None


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
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            model_file = ckpt.model_checkpoint_path
        else:
            raise ValueError(
                'No model file and could not find {}.'.format(model_dir))
    print('~~initialising from {}'.format(model_file), end='', flush=True)
    saver.restore(session, model_file)
    print('\r~~initialised~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def tokenise(sentence, vocab):
    """Splits up the sentence into words, removes unknowns etc.
    Returns both the rearranged sentence and the sentence as a list
    of int ids"""
    split_sentence = reuters.word_split(sentence)
    split_sentence = [word if word in vocab else '<UNK>'
                      for word in split_sentence]
    sentence_ids = [int(vocab[word]) for word in split_sentence]
    return ' '.join(split_sentence), sentence_ids


def get_vector(words, vocab_size=20000, model_dir='bigvocab_untied_models'):
    """Goes end to end from a string to the vector representation"""
    global _model  # I'm so sorry
    global _sess
    global _vocab
    if not _model:
        print('~~getting model')
        if not _vocab:
            _, _, _vocab = reuters.get_reuters(
                most_common=vocab_size)
        _model = load_encoder(_vocab, [10, 25, 50, 100, 200])
        _sess = tf.InteractiveSession()
        initialise(_sess, model_dir)
        print('~~done')
    new_words, word_ids = tokenise(words, _vocab)
    word_ids, bucket = _model.pad_and_bucket(
        [word_ids], reuters.get_special_ids())
    return _model.embed_batch(_sess, word_ids, bucket).flatten()


def main(_):
    global _vocab
    global _model

    print('~~loading data', end='', flush=True)
    x, y, _vocab = reuters.get_reuters(most_common=FLAGS.vocab_size)
    data = x+y  # just do it all
    data = [item[0] for item in data if len(item[0]) <= 200]
    print('\r~~done. ({} documents)'.format(len(data)))
    print('~~loading model', end='', flush=True)
    _model = load_encoder(_vocab, [10, 25, 50, 100, 200])
    print('\r~~done.        ')
    sess = tf.Session()
    with sess.as_default():
        initialise(sess, FLAGS.model_dir, FLAGS.model_file)
        vecs = []
        # This is kinda dumb (ie. slow)
        # should do more than one at a time
        # the reason we aren't is because I
        # can't be bothered making sure they
        # stay in the same order and I want
        # to make sure they all get pushed
        # through the appropriate bucket
        # because that was how the model
        # was trained.
        print('~~getting vectors')
        for i, seq in enumerate(data):
            padded_seq, bucket = _model.pad_and_bucket([seq], _vocab)
            vecs.append(_model.embed_batch(sess, padded_seq, bucket))
            if i % 10 == 0:
                print('\r~~  ({})'.format(i), end='', flush=True)
        print()
        print('~~writing to file (as giant numpy array)')
        print('lol gotcha I\'ve done nothing')
        print('bye')


if __name__ == '__main__':
    tf.app.run()
