from __future__ import print_function
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D,Activation
from keras.datasets import imdb

import matplotlib.pyplot as plt


class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def plot_loss(self, loss_type='epoch'):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc='upper right')
        plt.show()
        plt.savefig("cnn_loss_history.png")


# define callbacks
history = LossHistory()
early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=3, verbose=1, mode='auto')

max_features = 10000
maxlen = 400  # 每篇文章最多保留 200 个词
batch_size = 64

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 64, input_length=maxlen))  # 嵌入层
# 一维卷积，卷积核个数 128，卷积核维度 5
# 200 * 64 => (200 - 5 + 1) * 128 = (196, 128)
# 每个卷积核会产生一个 196 维的向量，128 个卷积核，也就是 128 个 196 维的向量
# 但是 LSTM 的第一步输入相当于是所有卷积核对前 5 个词的卷积结果，依次类推，所以 LSTM 一共是 196 步，每步的输入是一个 128 维的向量
# 由于是双向循环神经网络，每步的输出也是一个 128 维的向量，但是只在最后一步产生一个输出
model.add(Conv1D(activation="relu", filters=128, kernel_size=5, strides=1, padding="valid"))
model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
model.summary()

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=[x_test, y_test],
          callbacks=[history, early_stopping])

history.plot_loss('epoch')