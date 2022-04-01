#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile

import numpy as np
import pandas as pd
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tqdm import tqdm

import pickle

# from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard.plugins import projector
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
# In[2]:

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


count_save = 5



# In[3]:


zh_wiki_id = open("data/zh_wiki_id_w_d").readline()
word_to_id = pickle.load(open("data/word_to_id_w_d.pkl", "rb"))
id_to_word = pickle.load(open("data/id_to_word_w_d.pkl", "rb"))
# word_count = pickle.load(open("data/word_count_w_d.pkl", "rb"))
print("len(id_to_word) : ",len(id_to_word))
print("len(word_to_id) : ",len(word_to_id))
print("len(zh_wiki_id) : ",len(zh_wiki_id))
# import gensim
# from gensim.models import word2vec
# from gensim.similarities import WmdSimilarity
# ori_model = word2vec.Word2Vec.load('./word2vec.model')
# vector_dim = 300
# embedding_matrix = np.zeros((len(ori_model.wv.vocab),vector_dim),dtype='float32')
# for i in range (len(ori_model.wv.vocab)):
#     embedding_vector = ori_model.wv[ori_model.wv.index2word[i]]
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector


def getWord(data, num, data_index):
    sub_data_string = data[data_index:data_index+num*(6+1)]
    result = []
    for index, item in enumerate(sub_data_string.split()):
        if index == num: break
        data_index += len(item) + 1
        result.append(int(item))
    if len(result) < num:
        return getWord(data, num, 0)
    assert len(result) == num
    return result, data_index


def generate_batch(batch_size, skip_window, num_skips):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    np.set_printoptions(suppress=True)  # make numpy save long integer
    batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    assert batch_size >= span
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builti

    result, data_index = getWord(zh_wiki_id, span, data_index)
    buffer.extend(result)

    for i in range(batch_size):
        context_words = [w for w in range(span) if w != skip_window]
        batch[i, :] = [buffer[token] for idx, token in enumerate(context_words)]
        labels[i, 0] = buffer[skip_window]
        result, data_index = getWord(zh_wiki_id, 1, data_index)
        buffer.append(result[0])
        if data_index > len(zh_wiki_id):
            result, data_index = getWord(zh_wiki_id, span - 1, 0)
            buffer.extend(result)
        if i == batch_size - span:
            last_index = data_index

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = last_index
    return batch, labels

data_index = 0
batch, labels = generate_batch(batch_size=8, skip_window=1, num_skips=2*1)
for i in range(8):
    print(batch[i, 0], id_to_word[batch[i, 0]],
          batch[i, 1], id_to_word[batch[i, 1]],
          '->', labels[i, 0], id_to_word[labels[i, 0]])

batch_size = 100
# batch_size = 256
embedding_size = 300    # Dimension of the embedding vector.
skip_window = 5    # How many words to consider left and right.
num_skips = 2*skip_window    # How many times to reuse an input to generate a label.
num_sampled = 5    # Number of negative examples to sample.
# num_sampled = 128    # Number of negative examples to sample.
vocabulary_size = len(id_to_word)

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation.
# valid_size = 16    # Random set of words to evaluate similarity on.
# valid_window = 100    # Only pick dev samples in the head of the distribution.
# valid_examples = np.random.choice(valid_window, valid_size, replace=False)
valid_examples = list(range(1, 10))
valid_examples = list(range(50, 59))
valid_size = len(valid_examples)
graph = tf.Graph()

with graph.as_default():
    # Input data.
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size, num_skips])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/gpu:0'):
        # Look up embeddings for inputs.
        with tf.name_scope('embeddings'):
            # save_embedding = tf.constant(embedding_matrix) #new create
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            # embeddings = tf.Variable(initial_value=save_embedding,trainable=True) #new
            print("embeddings:", embeddings)
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            print("embed:", embed)
            # take mean of embeddings of context words for context embedding
        #             embed_context = tf.reduce_mean(embed, 1)

        with tf.name_scope('global-attention'):
            attention_size = embedding_size - 0
            attention_w = tf.Variable(tf.ones(shape=[2*embedding_size, attention_size], dtype=tf.float32))
            attention_b = tf.Variable(tf.zeros(shape=[attention_size], dtype=tf.float32))
            #new begin
            target_embed = tf.nn.embedding_lookup(embeddings, train_labels)
            target_embed = tf.tile(target_embed, multiples=[1, num_skips, 1])
            embed_concat = tf.concat([embed, target_embed], 2)
            print("embed_concat:", embed_concat)
            attention_matmul = tf.tanh(tf.tensordot(embed_concat, attention_w, axes=[[2], [0]]) + attention_b)
            #end
            # attention_matmul = tf.tanh(tf.tensordot(embed, attention_w, axes=[[2], [0]]) + attention_b)
            # B,W,D * D,A -> B,W,A
            print("attention_matmul:", attention_matmul)

            attention_w_a = tf.Variable(tf.ones(shape=[attention_size], dtype=tf.float32))
            attention = tf.nn.softmax(tf.tensordot(attention_matmul, attention_w_a, axes=[[2], [0]]))
            # B,W,A * A -> B,W
            print("attention:", attention)

            attention_context = tf.reduce_sum(tf.multiply(embed, tf.expand_dims(attention, -1)), axis=1)
            # B,W,D * B,W,1 -> B,W,D -> B,D
            print("attention_context:", attention_context)

            # statistics attention info
            attention_mean, attention_var = tf.nn.moments(attention, axes=[-1])
            attention_mean = tf.reduce_mean(attention_mean)
            attention_var = tf.reduce_mean(attention_var)
            print("attention_mean:", attention_mean)
            print("attention_var:", attention_var)

        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal(
                    [vocabulary_size, embedding_size],
                    stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # Explanation of the meaning of NCE loss:
    #     http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    with tf.name_scope('loss'):
        #         loss = tf.reduce_mean(
        #             tf.nn.nce_loss(nce_weights, nce_biases, embed_context, train_labels,
        #                            num_sampled, vocabulary_size))
        #         print(train_labels, embed_context)
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                #                         inputs=embed_context,
                inputs=attention_context,
                #                         labels=embed_context,
                #                         inputs=train_labels,
                num_sampled=num_sampled,
                num_classes=vocabulary_size))

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
        # optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        # optimizer = tf.compat.v1.train.AdagradOptimizer(0.1).minimize(loss)
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm

    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    print("valid_embeddings:", valid_embeddings)
    print("normalized_embeddings:", normalized_embeddings)
    print("similarity:", similarity)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()
    saver_embed = tf.train.Saver([embeddings])



num_steps = 200000001
# num_steps = 1

log_dir = "./log_dir/"
# log_result_dir = "./log_result_dir/"

tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True

with tf.Session(graph=graph, config=tfconfig) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(log_dir, session.graph)

    # We must initialize all variables before we use them.
    # init.run()
    #     saver = tf.train.import_meta_graph('./checkpoint_dir/MyModel-1000.meta')
    saver.restore(session, tf.train.latest_checkpoint(log_dir))
    # saver_embed.restore(session, tf.train.latest_checkpoint(log_dir))
    print('Initialized')
    count_num = 0
    average_loss = 0
    avetage_attention = 0
    std_attention = 0
    start_index = 0
    for step in xrange(start_index, num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, skip_window=skip_window, num_skips=num_skips)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # Define metadata variable.
        run_metadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
        # Feed metadata variable to session for visualizing the graph in TensorBoard.
        _, summary, loss_val, attention_val, std_val = session.run(
            [optimizer, merged, loss, attention_mean, attention_var],
            feed_dict=feed_dict,
            run_metadata=run_metadata)
        average_loss += loss_val
        avetage_attention += attention_val
        std_attention += std_val

        #         print("embed:", embed.eval(feed_dict=feed_dict))
        #         print("attention_w:", attention_w.eval())
        #         print("attention_matmul:", attention_matmul.eval(feed_dict=feed_dict))
        #         print("attention:", attention.eval(feed_dict=feed_dict))
        #         print("attention_context:", attention_context.eval(feed_dict=feed_dict))

        # Add returned summaries to writer in each step.
        writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, 'step%d' % step)

        if step % 2000 == 0:
            if step > 0 and step != start_index:
                average_loss /= 2000
                avetage_attention /= 2000
                std_attention /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step ', step, ': ', average_loss, ': ', data_index)
            print('Average attention at step ', step, ': ', avetage_attention)
            print('Variance attention at step ', step, ': ', std_attention)
            average_loss = 0
            avetage_attention = 0
            std_attention = 0
        #             print("attention:", attention.eval(feed_dict=feed_dict)[:2])
        #             print("attention:", attention.eval(feed_dict=feed_dict))

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            print('-----------------------')
            for i in xrange(valid_size):
                valid_word = id_to_word[valid_examples[i]]
                # valid_word = ori_model.wv.index2word[i]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = id_to_word[nearest[k]]
                    # close_word = ori_model.wv.index2word[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
                # print('-----------------------')

            # Save the model for checkpoints.
            # saver.save(session, os.path.join(log_dir, 'model.ckpt'), global_step=step)

        if step % 100000 == 0 and step != start_index:
            print('-----------------------')
            saver.save(session, os.path.join(log_dir, 'model.ckpt'))
            print("Save model ok,Great.d(^_^)b")
            print('-----------------------')
            word2vec = embeddings.eval()
            print(word2vec.shape, type(word2vec))
            #             np.save("result/004#cbow_self-attention_0521_"+str(step), word2vec)
            # saver.save(session, os.path.join(log_dir, 'model.ckpt'), global_step=step)
            # np.save(log_result_dir + "test.npy", word2vec)
            # np.savetxt(log_result_dir + "file.txt" , word2vec,fmt='%.23f')
        # if step % 10000000 == 0 and step != start_index:
        #     count_num = count_num + 1
        #     saver.save(session, os.path.join(log_dir,str(count_num),'kw_model.ckpt'))

    final_embeddings = normalized_embeddings.eval()

    # Write corresponding labels for the embeddings.
    # with open(log_dir + '/metadata.tsv', 'w',encoding='utf-8') as f:
    #     for i in xrange(vocabulary_size):
    #         f.write(id_to_word[i] + '\n')

    # Save the model for checkpoints.
    saver.save(session, os.path.join(log_dir, 'model.ckpt'), global_step=step)

    # Create a configuration for visualizing embeddings with the labels in TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    projector.visualize_embeddings(writer, config)

writer.close()
