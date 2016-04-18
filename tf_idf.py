"""This module is for constructing tf-idf representations of the documents
Just simple numpy implementation, designed to run once.
"""
import collections
import logging

import numpy as np


def tf_idf(documents, vocab=None, counts=None):
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

    Returns:
        numpy ndarray with shape `(len(documents), len(vocab))`.
    """
    if not counts:  # we'll have to gather them
        print('Counting: ')
        counts = collections.Counter()
        for document in documents:
            for symbol in document:
                print(symbol)
                counts[symbol] += 1
    total_symbols = sum(counts.values())
    # let's make an inverse document frequency vector so we can do a big
    # componentwise multiply at the end
    idf = np.zeros(len(vocab))
    for symbol in counts:
        if vocab:
            idf[vocab[symbol]] = total_symbols / counts[symbol]
        else:
            idf[symbol] = total_symbols / counts[symbol]
    idf = np.log(idf)
    # now we have document frequencies we need term frequencies per doc
    # this is going to be fairly large
    tfs = [_term_freqs(doc, vocab, len(counts)) for doc in documents]
    tfs = np.array(tfs)

    return tfs * idf


def _term_freqs(doc, vocab=None, num_symbols=None):
    """Calculate the term frequencies for a given document."""
    counts = collections.Counter(doc)
    if not num_symbols:
        num_symbols = len(vocab)
    vector = np.zeros(num_symbols)
    inverse_cardinality = 1.0 / len(num_symbols)
    for symbol in counts:  # only these guys get non zeros
        if vocab:
            vector[vocab[symbol]] = inverse_cardinality * counts[symbol]
        else:
            vector[symbol] = inverse_cardinality * counts[symbol]
    return vector


def main():
    """If run as a script, looks for data and if not found, generates it."""
    import reuters
    training, test, vocab = reuters.get_reuters()
    train_vectors = tf_idf(training)
    test_vectors = tf_idf(test)
    # write the data :)
    with open('train_tf-idf', 'w') as f:
        np.savetxt(f, train_vectors)
    with open('test_tf-idf', 'w') as f:
        np.savetxt(f, test_vectors)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
