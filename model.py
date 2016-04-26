"""This defines the neural network model used to construct
the document vectors"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import random

import numpy as np
import tensorflow as tf

# may as well experiment with the batch normed guy
import mrnn


class SequenceAutoencoder(object):
    """A sequence autoencoder.

    For now this is will be quite simple, but we will see.
    """

    def __init__(self, vocab_size, size, num_layers, batch_size,
                 buckets,
                 max_grad_norm=10000.0, learning_rate=0.01,
                 learning_rate_decay=0.99,
                 num_samples=512, train=True, dropout=1.0):
        """Create the model.

        Args:
            vocab_size (int): size of the vocabulary.
            size (int): number of units in each layer.
            num_layers (int): the number of layers in the model.
            batch_size (int): size of batches used for training.
                Can potentially be changed after training if desired.
            buckets: a list of maximum sequence lengths. Because the
                input is the output this should just be a list of ints.
                Should be sorted in ascending order.
            max_grad_norm (float): gradients will be clipped to have
                this norm.
            learning_rate (float): learning rate to start with.
            learning_rate_decay (float): how much to decay the learning
                rate when necessary.
            num_samples (int): number of samples for the sampled softmax.
            train (bool): whether or not to construct the extra ops for
                training.
            dropout (float): probability of keeping inputs to each layer.
        """
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay)
        self.global_step = tf.Variable(0, trainable=False)
        self.buckets = buckets = [(i, i) for i in buckets]  # to easier copy tf egs.

        # we need an output projection for sampled softmax
        output_projection = None
        softmax_loss_func = None
        # double check we want to do it
        if num_samples > 0 and num_samples < self.vocab_size:
            with tf.device('/cpu:0'):
                w = tf.get_variable('proj_w', [size, self.vocab_size])
                w_t = tf.transpose(w)
                b = tf.get_variable('proj_b', [self.vocab_size])
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                with tf.device('/cpu:0'):
                    labels = tf.reshape(labels, [-1, 1])
                    return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels,
                                                      num_samples, vocab_size)
            softmax_loss_func = sampled_loss

        # make the RNN cell
        cell = mrnn.IRNNCell(size, nonlinearity=tf.nn.relu, weightnorm='none')
        if dropout != 1.0:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.embedding_tied_rnn_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                self.vocab_size,
                size,
                output_projection=output_projection,
                feed_previous=do_decode)

        # feeds, lots of them
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # biggest bucket
            # set up the placeholders
            self.encoder_inputs.append(
                tf.placeholder(tf.int32, shape=[None],
                               name='encoder{}'.format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(
                tf.placeholder(tf.int32, shape=[None],
                               name='decoder{}'.format(i)))
            self.target_weights.append(
                tf.placeholder(tf.float32, shape=[None],
                               name='weight{}'.format(i)))

        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        # training outputs, losses
        if not train:
            self.outputs, self.losses = tf.nn.seq2se1.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_func)
            # project
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                        ]
        else:  # actually do the training
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_func)

        # gradients
        params = tf.trainable_variables()
        if train:
            self.gradient_norms = []
            self.updates = []
            # nb -- room to move on this
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_grad_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(
            tf.trainable_variables() + [self.global_step, self.learning_rate])

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
        """Run a step.

        Args:
            session: tensorflow session to run in.
            encoder_inputs: list of numpy int vectors to feed as encoder inputs.
            decoder_inputs: list of numpy int vectors to be decoder inputs
            target_weights: list of numpy float vectors for target weights.
            bucket_id: which bucket of the model to use.
            forward_only: whether to do the backward (training) step.

        Returns:
            A triple consisting of gradient norm (or None), average perplexity
            and outputs.

        Raises:
            ValueError: if the lengths of any of the lists are wrong.
        """
        # check sizes
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError('Encoder length must be equal to the bucket length')
        if len(decoder_inputs) != decoder_size:
            raise ValueError('Decoder length must be equal to the bucket length')
        if len(target_weights) != decoder_size:
            raise ValueError('Weights length must be equal to the bucket length')

        # set up the feed dict
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # output
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # sgd update for this bucket
                           self.gradient_norms[bucket_id],  # to have a look at
                           self.losses[bucket_id]]  # actual loss
        else:
            output_feed = [self.losses[bucket_id]]  # just loss
            for l in xrange(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])  # outs

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None
        else:
            return None, outputs[0], outputs[1:]  # outputs, no norm

    def get_batch(self, data, bucket_id, special_ids):
        """get a batch of random data for given bucket

        We have to fiddle with the data to get it in the right shape.

        Args:
            data: tuple of size len(self.buckets) in which each element
                contains lists of pairs of input and output data that can be used
                to create a batch.
            bucket_id: integer -- the bucket we want a batch for.
            special_ids: dict of string -> int, containing the special ids we
                use for padding and starting. In particular `<GO>` should map
                to the id used to start a sequence and `<PAD>` to the padding
                id.

        Returns:
            triple (encoder_inputs, decoder_inputs, target_weights) for the
            constructed batch that is ready to be sent in to step(_).
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        # grab batch, maybe pad, add GO
        # could try reversing them, see how it goes
        for _ in xrange(self.batch_size):
            # not sure how I feel about choosing the batch with replacement...
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # pad the inputs
            # should test whether it is better to pad the front or back
            # but for now we pad the front to keep the sequence and outputs
            # close
            encoder_pad = [special_ids['<PAD>']] * (encoder_size - len(encoder_input))
            encoder_inputs.append(encoder_pad + encoder_input)

            # decoder inputs -- put a <GO> and then pad the back
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([special_ids['<GO>']] + decoder_input +
                                  ([special_ids['<PAD>']] * decoder_pad_size))

        # now we reshape (although I feel like a better place to do this would
        # in the code that actually puts together the data)
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # for decoder inputs we need to make weights as well
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in xrange(self.batch_size)], dtype=np.int32))
            # target weights 0 for padding
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == special_ids['<PAD>']:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
