"""Tools for dealing with Reuters-21578 dataset

Contains methods for downloading and parsing the data as well as setting up
tensorflow graph elements to use it.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import xrange

import os
import tarfile
import re
import xml.etree.ElementTree as ET
import collections
import logging
import math

import requests

import numpy as np
import tensorflow as tf


def _maybe_download(data_path='reuters21578.tar.gz'):
    """Check if it exists"""
    if not os.path.exists(data_path):
        # better grab it
        logging.info('Data not found, downloading')
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz'
        req = requests.get(url, stream=True)
        with open(data_path, 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:  # might want to ignore keep-alives
                    f.write(chunk)
        # got it
    # quick validate
    size = os.path.getsize(data_path) >> 20
    if size != 7:  # oh no!
        raise ValueError('data file is wrong size ({})'.format(size))
    return data_path


def _get_vocab_filename(level):
    """Gets a filename for the vocab file

    Args:
        level: whether we are interested in word or character

    Returns:
        str: the filename.
    """
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'vocab-{}.txt'.format(level))


def _read_one_file(filename, vocab_freqs, token_func, everything=True):
    """Reads one whole file and pulls out the data we need.
    Returns a list of lists looking like [text, label, set, id] where
    set is one of {train, test} according to `ModHayes` split.
    Adds to counts, text is tokenised using provided function.
    """
    # we have to get weird to make this all happy
    with open(filename, errors='ignore') as raw_file:
        file_str = raw_file.read()
        # skip some crap
        file_str = re.sub(r'&#\d{1,2};', '', file_str)
        # insert root tags
        file_str = '<root>' + file_str[file_str.find('\n'):-1] + '</root>'
        logging.info('..parsing %s', filename)
        root = ET.fromstring(file_str)
        data = []
        for child in root:
            if child.attrib['TOPICS'] == 'YES' or everything:
                try:
                    text = child.find('./TEXT/BODY').text
                except AttributeError:
                    # some don't have body tags
                    text = child.find('./TEXT').text
                text = token_func(text)
                if vocab_freqs is not None:
                    for symbol in text:
                        vocab_freqs[symbol] += 1
                topics = [bytes(d.text, 'utf-8')
                          for d in child.findall('./TOPICS/D')]
                if child.attrib['CGISPLIT'] == 'TRAINING-SET':
                    dataset = 'train'
                else:
                    dataset = 'test'
                data.append(
                    [text, topics, dataset, child.attrib['NEWID']])
        return data


def word_split(text):
    """This is our main tokeniser, could do stemming etc, does not."""
    # first remove caps
    text = text.casefold()
    # replace numbers, including with commas or .
    text = re.sub(r'(\d+([.,]?))+', ' <NUMBER> ', text)
    text = re.sub(r'[.?!]', ' <STOP> ', text)  # mark ends of sentences
    # remove remaining punctuation
    text = re.sub(r'[^\w\s<>]', ' <PUNCT> ', text)
    return text.split()  # finally split on whitespace


def char_split(text):
    """Split it character wise"""
    return list(text)


def _log_stats(data):
    """Just log some things about the data"""
    cumulative = 0
    longest = -1
    shortest = 1000000
    for item in data:
        length = len(item)
        cumulative += length
        if length > longest:
            longest = length
        if length < shortest:
            shortest = length
    mean = cumulative/len(data)
    cumulative = 0
    for item in data:
        cumulative += (mean - len(item))**2
    stddev = math.sqrt(cumulative / len(data))
    logging.info('..mean: %d (stddev: %d)', mean, stddev)
    logging.info('..max: %d', longest)
    logging.info('..min: %d', shortest)


def get_special_ids():
    """returns a dict of the special ids"""
    return {
        '<GO>': 0,
        '<UNK>': 1,
        '<STOP>': 2,
        '<PUNCT>': 3,
        '<PAD>': 4
    }


def get_text_and_labels(data_dir='data', ids=None, split='both'):
    """Gets the dataset just as raw text and labels.

    Args:
       data_dir: where the data should be. If it is not there it will be
            downloaded.
       ids: optional list of ids. If present only the matching records are
           returned.
       set: optional string, one of `train`, `test` or `both`.
    """
    if not os.path.exists(data_dir):
        with tarfile.open(_maybe_download(), 'r:gz') as datafile:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner) 
                
            
            safe_extract(datafile, path=data_dir)
    filenames = [os.path.join(data_dir, f)
                 for f in os.listdir(data_dir) if re.search('sgm$', f)]
    data = []
    for filename in filenames:
        data.extend(_read_one_file(filename, None, word_split))
    # check the split while we still can
    if split != 'both':
        data = [item for item in data if item[2] == split]
    if ids:
        text = [' '.join(item[0]) for item in data if item[-1] in ids]
        labels = [item[1] for item in data if item[-1] in ids]
    else:
        text = [' '.join(item[0]) for item in data]
        labels = [item[1] for item in data]

    return text, [[label.decode()
                   for label in labelling]
                  for labelling in labels]


def get_reuters(data_dir='data', level='word', min_reps=1, most_common=10000):
    """Get the dataset as (training, test, vocab).
    first two are tuples containing a sequence, the labels and what part
    split they are in, vocab is
    a map of strings->int ids.
    """
    if not os.path.exists(data_dir):
        with tarfile.open(_maybe_download(), 'r:gz') as datafile:
            logging.info('extracting archive')
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner) 
                
            
            safe_extract(datafile, path=data_dir)
    # we are just going to hold the data in memory
    filenames = [os.path.join(data_dir, f)
                 for f in os.listdir(data_dir) if re.search('sgm$', f)]
    all_data = []
    vocab_freqs = collections.Counter()
    split_func = word_split if level == 'word' else char_split
    for filename in filenames:
        all_data.extend(_read_one_file(filename, vocab_freqs, split_func))
    logging.info('got %d records', len(all_data))
    ordered_words = vocab_freqs.most_common()
    to_remove = set()
    if min_reps > 1:
        for word, count in ordered_words[-1:0:-1]:
            # loop backwards
            if count > min_reps:
                break  # must have them all
            to_remove.add(word)
            del vocab_freqs[word]
    if most_common is not None:
        for word, count in ordered_words[most_common:]:
            to_remove.add(word)
            del vocab_freqs[word]
    logging.info('Removing %d words from vocab.', len(to_remove))
    for item in all_data:
        item[0] = ['<UNK>' if i in to_remove else i for i in item[0]]
    symbol_to_id = get_special_ids()
    id_num = 5  # start after specials
    for symbol in vocab_freqs:
        if symbol not in symbol_to_id:
            symbol_to_id[symbol] = id_num
            id_num += 1
    logging.info('..last id %d', id_num)
    logging.info('..vocab size %d', len(symbol_to_id))

    for item in all_data:
        item[0] = [symbol_to_id[i] for i in item[0]]
    training = [item for item in all_data if item[2] == 'train']
    test = [item for item in all_data if item[2] != 'train']

    vocab_filename = _get_vocab_filename(level)
    if not os.path.exists(vocab_filename):
        logging.info('..no existing vocab file, writing.')
        _write_vocab(symbol_to_id, vocab_filename)
        logging.info('..written to "%s"', vocab_filename)

    return (training, test, symbol_to_id)


def get_vocab():
    """Gets the words if the file is there"""
    with open(_get_vocab_filename('word')) as f:
        return {a: int(b) for a, b in (line.split(',') for line in f)}


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _write_vocab(vocab, path):
    """Writes vocab dict as csv"""
    with open(path, 'w') as f:
        for key in vocab:
            f.write('{},{}\n'.format(key, vocab[key]))


def check_tf_data_exists(data_dir='data/tf-records', records_per=500):
    """Double checks the data exists in Tensorflow Record format"""
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        logging.info('No tf record data, fixing')
        training, test, vocab = get_reuters()
        _write_vocab(vocab, os.path.join(data_dir,))
        filename = os.path.join(data_dir, 'train-data')
        logging.info('writing training set')
        _write_tfrecords(training, filename, records_per)
        logging.info('writing test set')
        filename = os.path.join(data_dir, 'test-data')
        _write_tfrecords(test, filename, records_per)
    else:
        logging.info('Found the data :)')


def _write_tfrecords(data, prefix, records_per=500, sort_length=True):
    """write one set"""
    if sort_length:
        data.sort(key=len)
    num_files = ((len(data)-1) // records_per) + 1
    item_iter = iter(data)
    logging.info('..this will take %d files', num_files)
    filename = prefix + '-{}.tfrecords'
    for filenum in xrange(num_files):
        f = tf.python_io.TFRecordWriter(filename.format(filenum))
        for i in xrange(records_per):
            try:
                item = next(item_iter)
            except StopIteration:
                break
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'text': _int64_feature(item[0]),
                    'labels': _bytes_feature(item[1]),
                    'id': _int64_feature([int(item[3])])
                    }))
            f.write(example.SerializeToString())
        f.close()


def get_input_pipeline(data_dir, train, num_epochs, batch_size, buckets):
    """Gets a full input pipeline of file readers, shuffling etc.
    Returns (data, labels, bucket_idx) batches where data is
    [[batch_size] * bucket_size] for whatever bucket."""
    def read_file(filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_eg = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_eg,
            features={
                'text': tf.VarLenFeature([], tf.int64),
                'labels': tf.VarLenFeature([], tf.string),
            })
        text = [tf.cast(num, tf.int32) for num in features['text']]

        return text, features['text']

    if not num_epochs:
        num_epochs = None  # if unspecified loop forevererer

    if train:
        file_expr = os.path.join(data_dir, 'train-data-*')
    else:
        file_expr = os.path.join(data_dir, 'test-data-*')
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            tf.train.match_filenames_once(file_expr),
            num_epochs=num_epochs,
            shuffle=True)
        example, label = read_file(filename_queue)
        min_after_dequeue = 5000
        capacity = min_after_dequeue + 4 * batch_size
        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, num_threads=3)
        return example_batch, label_batch


def _run():
    """Potentially download and convert the data"""
    check_tf_data_exists()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    _run()
