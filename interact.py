"""Interacts with the trained model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import readline

import numpy as np
import tensorflow as tf

import model as sa
from reuters import get_reuters, word_split  # for the vocab
import reuters


flags = tf.app.flags
flags.DEFINE_string("model_dir", "models", "where to look for the latest")
flags.DEFINE_string("model_file", "", "if there is a specific file to use")
flags.DEFINE_integer("vocab_size", 5000, "how many words to use")
flags.DEFINE_integer("size", 256, "of each layer in the model")
flags.DEFINE_integer("num_layers", 2, "recurrent layers in the model")

FLAGS = flags.FLAGS

_buckets = [10, 25, 50, 100, 200]


def load_model(vocab_size=5000, size=256, num_layers=2):
    """Loads the model (without training ops) and returns it"""
    _, _, vocab = get_reuters(most_common=vocab_size)
    vocab_size = len(vocab)
    inv_vocab = {b: a for a, b in vocab.items()}
    print('~~Creating model', end='', flush=True)
    model = sa.SequenceAutoencoder(
        vocab_size, size, num_layers, 1, _buckets,
        train=False)
    print('\r~~Got model.    ', flush=True)
    return model, vocab, inv_vocab


def initialise(session, model, model_dir, model_file=""):
    """Initialise the model either from a directory
    (in which case we use the latest checkpoint) or a
    specific file."""
    if not model_file:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            model_file = ckpt.model_checkpoint_path
        else:
            raise ValueError('No model file specified and could not find'
                             ' checkpoint.')
    print('~~initialising from {}'.format(model_file), end='', flush=True)
    model.saver.restore(session, model_file)
    print('\r~~initialised~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def tokenise(sentence, vocab):
    """Splits up the sentence into words, removes unknowns etc.
    Returns both the rearranged sentence and the sentence as a list
    of int ids"""
    split_sentence = word_split(sentence)
    split_sentence = [word if word in vocab else '<UNK>'
                      for word in split_sentence]
    sentence_ids = [int(vocab[word]) for word in split_sentence]
    return ' '.join(split_sentence), sentence_ids


def find_bucket(data):
    """gets the appropriate bucket number"""
    for bucket_id, bucket_size in enumerate(_buckets):
        if len(data) < bucket_size:
            return bucket_id
    return -1


def pad(data, bucket_id):
    """pads the data to fit the given bucket.

    Returns:
        (encoder_inputs, decoder_inputs, target_weights): everything
            required to feed into SequenceAutoencoder.step().
    """
    special_ids = reuters.get_special_ids()
    bucket_size = _buckets[bucket_id]
    encoder_pad = [special_ids['<PAD>']] * (bucket_size - len(data))
    encoder_input = encoder_pad + data
    encoder_input = np.array(encoder_input, dtype=np.int32).reshape((-1, 1))
    # only the first one matters here
    decoder_input = [special_ids['<GO>']] * bucket_size
    decoder_input = np.array(decoder_input, dtype=np.int32).reshape((-1, 1))
    # set up the weights, not a big deal though (just for the loss calculation)
    weights = np.ones(bucket_size, dtype=np.float32)
    weights[len(data):] = 0
    weights = weights.reshape((-1, 1))
    return encoder_input, decoder_input, weights


def encode_decode(session, model, input_data):
    """Run the encoder and the decoder on the sentence, see what
    we get."""
    bucket = find_bucket(input_data)
    if bucket == -1:
        raise ValueError('sentence too long :( try again with longer buckets')
    print('~~~~using bucket {}'.format(bucket))
    encoder_inputs, decoder_inputs, weights = pad(input_data, bucket)
    results = model.step(session, encoder_inputs, decoder_inputs, weights,
                         bucket, True)
    return results


def safe_sample(pvals):
    """Gets a sample, defaulting to argmax when numerical issues arise.
    """
    try:
        exppvals = np.exp(pvals)
        return np.argmax(np.random.multinomial(1, exppvals/exppvals.sum()))
    except:
        return np.argmax(pvals)


def main(_):
    """Loads up the model (fails if it can't find an appropriate file)
    and drops into a loop taking input and pushing it through the model."""
    model, vocab, inv_vocab = load_model(vocab_size=FLAGS.vocab_size,
                                         size=FLAGS.size,
                                         num_layers=FLAGS.num_layers)
    sess = tf.Session()
    with sess.as_default():
        initialise(sess, model, model_dir=FLAGS.model_dir,
                   model_file=FLAGS.model_file)
        try:
            while True:
                sentence = input('>')
                sentence, sentence_ids = tokenise(sentence, vocab)
                print('~~Input:')
                print('~~{}'.format(sentence))
                # now we have to run the model on it
                result = encode_decode(sess, model, sentence_ids)
                print('~~~~~~')
                print('~~Result: ')
                print(' '.join(
                    [inv_vocab[int(safe_sample(r[0]))]
                     for r in result[2][:len(sentence_ids)]]))
                print('~~~~(loss: {})'.format(result[1]))
                print('~~~~~~')

        except (KeyboardInterrupt, EOFError):
            print('\nbye')


if __name__ == '__main__':
    tf.app.run()
