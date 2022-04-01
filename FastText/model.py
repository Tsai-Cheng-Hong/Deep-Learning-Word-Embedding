# coding: utf-8
from gensim.models import word2vec, fasttext

# Train
train_data = word2vec.LineSentence('wiki_text_seg.txt')
model = fasttext.FastText(train_data,size=300)
model.save('fasttext.model')
