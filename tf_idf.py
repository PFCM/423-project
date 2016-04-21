"""This module is for constructing tf-idf representations of the documents
Just simple numpy implementation, designed to run once.
"""
import collections
import logging
import os

import numpy as np


def tf_idf(documents, vocab=None, counts=None, sequence_ids=True):
    """Converts a set of documents to tf-idf weighted vectors of length
    using the indices defined in `vocab`.

    Args:
        documents: A list of documents. Each document is expected to be a
            list of symbols.
        vocab: a dictionary of symbol->int, where the numbers represent the
            index of the word's weight in the final vector. If not specified,
            (default) we assume that documents is in fact already a list of
            ints.
        counts: a dictionary of symbol->count across the whole corpus. If not
            provided, will be generated.
        sequence_ids: whether the sequences have already been transformed via
            `vocab`.

    Returns:
        numpy ndarray with shape `(len(documents), len(vocab))`.
    """
    if not counts:  # we'll have to gather them
        print('Counting: ')
        counts = collections.Counter()
        for document in documents:
            for symbol in document:
                counts[symbol] += 1
    total_symbols = sum(counts.values())
    # let's make an inverse document frequency vector so we can do a big
    # componentwise multiply at the end
    vocab_size = len(counts) if not vocab else len(vocab)
    idf = np.zeros(vocab_size)
    logging.info('TF-IDF will have %d dimensions', vocab_size)
    for symbol in counts:
        if vocab and not sequence_ids:
            idf[vocab[symbol]] = total_symbols / counts[symbol]
        else:
            idf[symbol] = total_symbols / counts[symbol]
    idf = np.log(idf)
    # now we have document frequencies we need term frequencies per doc
    # this is going to be fairly large
    tfs = [_term_freqs(doc, None, vocab_size) for doc in documents]
    tfs = np.array(tfs)

    return tfs * idf


def _term_freqs(doc, vocab=None, num_symbols=None):
    """Calculate the term frequencies for a given document."""
    counts = collections.Counter(doc)
    if not num_symbols:
        num_symbols = len(vocab)
    vector = np.zeros(num_symbols)
    inverse_cardinality = 1.0 / num_symbols
    for symbol in counts:  # only these guys get non zeros
        if vocab:
            vector[vocab[symbol]] = inverse_cardinality * counts[symbol]
        else:
            vector[symbol] = inverse_cardinality * counts[symbol]
    return vector


def get_tf_idf():
    """Looks for data, opens and returns."""
    import reuters
    if os.path.exists('train_tf-idf.txt.gz') and \
       os.path.exists('test_tf-idf.txt.gz'):
        training = np.loadtxt('train_tf-idf.txt.gz')
        test = np.loadtxt('test_tf-idf.txt.gz')
        vocab = reuters.get_vocab()
    return training, test, vocab


def main():
    """If run as a script, looks for data and if not found, generates it."""
    import reuters
    training, test, vocab = reuters.get_reuters()
    train_vectors = tf_idf([item[0] for item in training], vocab)
    test_vectors = tf_idf([item[0] for item in test], vocab)
    # we should write the category labels as well somehow??
    # write the data :)
    np.savetxt('train_tf-idf.txt.gz', train_vectors)
    np.savetxt('test_tf-idf.txt.gz', test_vectors)

    # log an example to make sure it makes sense
    inv_vocab = {b: a for a, b in vocab.items()}
    logging.info('test:')
    logging.info('%s', ' '.join([inv_vocab[index] for index in training[0][0]]))
    logging.info('Word: %s has weight %.4f',
                 inv_vocab[training[0][0][0]],
                 train_vectors[0][training[0][0][0]])  # lots of square brackets.


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
