from gensim.models import Word2Vec
# numpy
import numpy as np

# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import logging
import sys

log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)
with open('./imdb_data/pos.txt', 'r', encoding='utf-8') as infile:
    pos_reviews = []
    line = infile.readline()
    while line:
        pos_reviews.append(line)
        line = infile.readline()

with open('./imdb_data/neg.txt', 'r', encoding='utf-8') as infile:
    neg_reviews = []
    line = infile.readline()
    while line:
        neg_reviews.append(line)
        line = infile.readline()

with open('./imdb_data/unsup.txt', 'r', encoding='utf-8') as infile:
    unsup_reviews = []
    line = infile.readline()
    while line:
        unsup_reviews.append(line)
        line = infile.readline()
# 1 代表积极情绪，0 代表消极情绪
y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))

x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)
import gensim


def labelizeReviews(reviews):
    print(len(reviews))
    for i, v in enumerate(reviews):
        yield gensim.utils.simple_preprocess(v, max_len=100)


x_train_tag = list(labelizeReviews(x_train))
x_test_tag = list(labelizeReviews(x_test))
unsup_reviews_tag = list(labelizeReviews(unsup_reviews))
size = 100

# 实例化 DM 和 DBOW 模型
log.info('D2V')
model = Word2Vec(size=200, window=10, min_count=1)
# 对所有评论创建词汇表
alldata = x_train_tag
alldata.extend(x_test_tag)
alldata.extend(unsup_reviews_tag)
model.build_vocab(alldata)
import random


def sentences_perm(sentences):
    shuffled = list(sentences)
    random.shuffle(shuffled)
    return (shuffled)


log.info('Epoch')
for epoch in range(10):
    log.info('EPOCH: {}'.format(epoch))
    model.train(sentences_perm(alldata), total_examples=model.corpus_count, epochs=1)
# 对训练数据集创建词向量，接着进行比例缩放（scale）。
size = 200


def buildWordVector(text):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word]
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


train_vecs = np.concatenate([buildWordVector(gensim.utils.simple_preprocess(z, max_len=200)) for z in x_train])
train_vecs = scale(train_vecs)
test_vecs = np.concatenate([buildWordVector(gensim.utils.simple_preprocess(z, max_len=200)) for z in x_test])
test_vecs = scale(test_vecs)
classifier = LogisticRegression()
classifier.fit(train_vecs, y_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

log.info(classifier.score(test_vecs, y_test))
