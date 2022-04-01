# coding: utf-8
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
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from tqdm import tqdm
import time
# from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard.plugins import projector
import sys,os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with open(filename,encoding='utf-8') as f:
        data = f.readlines()
    return data


data = read_data("./data/wiki_ckip_No_En.txt")
vocabulary = []
for line in tqdm(data):
    line = line.strip()
    for item in line.split():
#         print(item)
        vocabulary.append(item)
#     break
# vocabulary = [v for v in vocabulary]
print('Data size', len(vocabulary))


print(data[-1])
del data
vocabulary[-1]

vocabulary[:10]

print("len(set(vocabulary)) :",len(set(vocabulary)))

word_cnt = collections.Counter(vocabulary).most_common(len(set(vocabulary)))
print("word_cnt[-10:] :",word_cnt[-10:])

for index, (k,v) in enumerate(word_cnt):
    if v <= 25:
        print(index)
        break


import pickle
with open("data/word_count_w_d.pkl", "wb") as f:
    pickle.dump(word_cnt, f)

look_num = -2740
word_cnt[:look_num][-5:]

# vocabulary_size = 504175
vocabulary_size = 510000 - 2740
# vocabulary_size = len(vocabulary)
def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

print(count[-10:])

with open("data/zh_wiki_id_w_d", "w",encoding='utf-8') as f:
    context = [str(i) for i in data]
    f.write(" ".join(context))

import pickle
with open("data/word_to_id_w_d.pkl", "wb") as f:
    pickle.dump(dictionary, f)
with open("data/id_to_word_w_d.pkl", "wb") as f:
    pickle.dump(reverse_dictionary, f)
with open("data/word_count_w_d.pkl", "wb") as f:
    pickle.dump(count, f)

