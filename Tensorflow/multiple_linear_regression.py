import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

class multi_linear_regression():
    def __init__(self):
        self.w = tf.Variable(tf.random.normal([5, 1], dtype=tf.double), name='weight')
        self.b = tf.Variable(tf.zeros([1], dtype=tf.double), name='bias')
        self.epochs= 100
        self.learning_rate = 0.01
    
    def train_batch(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            hypothesis = tf.matmul(batch_x, self.w) + self.b
            loss = tf.reduce_mean(tf.square(hypothesis - batch_y))
            loss_w, loss_b = tape.gradient(loss, [self.w, self.b])
        self.w.assign_sub(loss_w * self.learning_rate)
        self.b.assign_sub(loss_b * self.learning_rate)
        return loss
    
    def train(self, x, y):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=50).batch(10)
        loss_mem = []
        for e in range(self.epochs):
            for (x, y) in dataset:
                loss = self.train_batch(x, y)
            loss_mem.append(loss)
        return loss_mem
    
    def predict(self, x, y):
        y_hat = tf.matmul(x, self.w) + self.b
        mse = tf.reduce_mean(tf.square(y_hat - y))
        rmse = tf.sqrt(mse)
        return rmse

N = 100
x, y = make_regression(n_samples=N, n_features=5, bias=10.0, noise=10.0, random_state=1)
y = np.expand_dims(y, axis=1)

train_N = int(N * 0.8)
train_x = x[:train_N]
train_y = y[:train_N]
test_x = x[train_N:]
test_y = y[train_N:]

model = multi_linear_regression()
loss_mem = model.train(train_x, train_y)
x_epoch = list(range(len(loss_mem)))

plt.plot(x_epoch, loss_mem)
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.show()

print(model.predict(test_x, test_y).numpy())

