# -*- coding: utf-8 -*-

"""
@Author  : captain
@time    : 18-7-6 下午12:56
@ide     : PyCharm  
"""
from __future__ import absolute_import
from __future__ import print_function

from data_utils_sentihood import *
from vocab_processor import *
from sklearn import metrics
from delayed_entnet_sentihood import Delayed_EntNet_Sentihood
from itertools import chain
from six.moves import range
from collections import defaultdict

import tensorflow as tf
import numpy as np

import sys
import random
import logging
import cPickle as pickle

import pprint

pp = pprint.PrettyPrinter()

tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate for the optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 128, "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 2, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("sentence_len", 50, "Maximum len of sentence.")
tf.flags.DEFINE_string("task", "Sentihood", "Sentihood")
tf.flags.DEFINE_integer("random_state", 67, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/sentihood/", "Directory containing Sentihood data")
tf.flags.DEFINE_string("opt", "ftrl", "Optimizer [ftrl]")
tf.flags.DEFINE_string("embedding_file_path", "data/glove.6B.300d.txt", "Embedding file path [None]")
tf.flags.DEFINE_boolean("update_embeddings", False, "Update embeddings [False]")
tf.flags.DEFINE_boolean("case_folding", True, "Case folding [True]")
tf.flags.DEFINE_integer("n_cpus", 6, "N CPUs [6]")
tf.flags.DEFINE_integer("n_keys", 7, "Number of keys [7]")
tf.flags.DEFINE_integer("n_tied", 2, "Number of tied keys [2]")
tf.flags.DEFINE_float("entnet_input_keep_prob", 0.8, "entnet input keep prob [0.8]")
tf.flags.DEFINE_float("entnet_output_keep_prob", 1.0, "entnet output keep prob [1.0]")
tf.flags.DEFINE_float("entnet_state_keep_prob", 1.0, "entnet state keep prob [1.0]")
tf.flags.DEFINE_float("final_layer_keep_prob", 0.8, "final layer keep prob [0.8]")
tf.flags.DEFINE_float("l2_final_layer", 1e-3, "Lambda L2 final layer [1e-3]")

FLAGS = tf.flags.FLAGS

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=FLAGS.n_cpus,
    inter_op_parallelism_threads=FLAGS.n_cpus,
)

# 只考虑4个top aspect
aspect2idx = {
    'general': 0,
    'price': 1,
    'transit-location': 2,
    'safety': 3,
}

assert FLAGS.n_keys >= 2
assert FLAGS.n_tied == 2

with tf.Session(config=session_conf) as sess:
    np.random.seed(FLAGS.random_state)
    tf.set_random_seed(FLAGS.random_state)  # 设置随机数种子便于复现
    batch_size = FLAGS.batch_size

    global_step = None
    optimizer = None

    if FLAGS.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
    elif FLAGS.opt == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate=FLAGS.learning_rate
        )

    model = Delayed_EntNet_Sentihood(
        batch_size,
        vocab_size,
        max_target_len,
        max_aspect_len,
        sentence_len,
        answer_size,
        embedding_size,
        session=sess,
        embedding_mat=word_vocab.embeddings,
        update_embeddings=FLAGS.update_embeddings,
        n_keys=FLAGS.n_keys,
        tied_keys=target_terms,
        l2_final_layer=FLAGS.l2_final_layer,
        max_grad_norm=FLAGS.max_grad_norm,
        optimizer=optimizer,
        global_step=global_step
    )
