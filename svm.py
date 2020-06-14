from gensim.utils import open, simple_preprocess
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random

# 读取影评内容
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

# 数据划分
# 1 代表积极情绪，0 代表消极情绪
y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)


# 创建TaggedDocument对象
# Gensim 的 Doc2Vec 工具要求每个文档/段落包含一个与之关联的标签。我们利用 TaggedDocument进行处理。
# 格式形如 “TRAIN_i” 或者 “TEST_i”，其中 “i” 是索引
def labelize_reviews(reviews, label_type):
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        yield TaggedDocument(simple_preprocess(v, max_len=100), [label])


x_train_tag = list(labelize_reviews(x_train, 'train'))
x_test_tag = list(labelize_reviews(x_test, 'test'))
unsup_reviews_tag = list(labelize_reviews(unsup_reviews, 'unsup'))

# 实例化Doc2vec模型
# 下面我们实例化两个 Doc2Vec 模型，DM 和 DBOW。
# gensim 文档建议多次训练数据，并且在每一步（pass）调节学习率（learning rate）或者用随机顺序输入文本。
# 接着我们收集了通过模型训练后的电影评论向量。
# DM 和 DBOW会进行向量叠加，这是因为两个向量叠加后可以获得更好的结果
size = 100
model_dm = Doc2Vec(min_count=1, window=10, vector_size=size, sample=1e-3, negative=5, workers=3,
                   epochs=10)
model_dbow = Doc2Vec(min_count=1, window=10, vector_size=size, sample=1e-3, negative=5, dm=0, workers=3,
                     epochs=10)

# 对所有评论创建词汇表
all_data = x_train_tag
all_data.extend(x_test_tag)
all_data.extend(unsup_reviews_tag)
model_dm.build_vocab(all_data)
model_dbow.build_vocab(all_data)


def sentences_perm(sentences):
    shuffled = list(sentences)
    random.shuffle(shuffled)
    return (shuffled)


for epoch in range(10):
    print('EPOCH: {}'.format(epoch))
    model_dm.train(sentences_perm(all_data), total_examples=model_dm.corpus_count, epochs=1)
    model_dbow.train(sentences_perm(all_data), total_examples=model_dbow.corpus_count, epochs=1)

# 获取生成的向量
# 获取向量有两种方式，一种是根据上面我们定义的标签来获取，另一种通过输入一篇文章的内容来获取这篇文章的向量。
# 更推荐使用第一种方式来获取向量。

# 第一种方法
train_arrays_dm = np.zeros((len(x_train), 100))
train_arrays_dbow = np.zeros((len(x_train), 100))
for i in range(len(x_train)):
    tag = 'train_' + str(i)
    train_arrays_dm[i] = model_dm.docvecs[tag]
    train_arrays_dbow[i] = model_dbow.docvecs[tag]
train_arrays = np.hstack((train_arrays_dm, train_arrays_dbow))
test_arrays_dm = np.zeros((len(x_test), 100))
test_arrays_dbow = np.zeros((len(x_test), 100))
for i in range(len(x_test)):
    tag = 'test_' + str(i)
    test_arrays_dm[i] = model_dm.docvecs[tag]
    test_arrays_dbow[i] = model_dbow.docvecs[tag]
test_arrays = np.hstack((test_arrays_dm, test_arrays_dbow))


# 第二种
def get_vecs(model, corpus):
    vecs = []
    for i in corpus:
        vec = model.infer_vector(simple_preprocess(i, max_len=300))
        vecs.append(vec)
    return vecs


train_vecs_dm = get_vecs(model_dm, x_train)
train_vecs_dbow = get_vecs(model_dbow, x_train)
train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

# 预测
classifier = LogisticRegression()
classifier.fit(train_arrays, y_train)
print(classifier.score(test_arrays, y_test))
y_prob = classifier.predict_proba(test_arrays)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.show()