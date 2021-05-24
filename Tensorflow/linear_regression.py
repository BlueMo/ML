import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

class linear_regression():
    def __init__(self):
        self.w = tf.Variable(tf.random.normal([1], dtype=tf.double), name='weight')
        self.b = tf.Variable(tf.zeros([1], dtype=tf.double), name='bias')
        self.epochs = 100
        self.learning_rate = 0.01
        print(self.w, self.b)
    
    def train_batch(self, batch_x, batch_y):
        with tf.GradientTape() as tape:
            hypothesis = batch_x * self.w + self.b
            loss = tf.reduce_mean(tf.square(hypothesis - batch_y))
            loss_w, loss_b = tape.gradient(loss, [self.w, self.b])
        self.w.assign_sub(self.learning_rate * loss_w)
        self.b.assign_sub(self.learning_rate * loss_b)
        return loss
    
    def train(self, train_x, train_y):
        dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        dataset = dataset.shuffle(buffer_size=50).batch(10)

        loss_mem = []
        for e in range(self.epochs):
            for (x,y) in dataset:
                loss = self.train_batch(x, y)
            loss_mem.append(loss)
        return loss_mem
    
    def test(self, target_x):
        res = target_x * self.w + self.b
        return res
    
    def predict(self, test_x, test_y):
        y_hat = test_x * self.w + self.b
        error = y_hat - test_y
        mse = np.mean(error * error)
        rmse = np.sqrt(mse)
        return rmse

x, y = make_regression(n_samples=100, n_features=1, bias=10.0, noise=10.0, random_state=1)
y = np.expand_dims(y, axis=1)

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]

model = linear_regression()
loss_mem = model.train(train_x, train_y)
x_epoch = list(range(len(loss_mem)))

plt.plot(x_epoch, loss_mem)
plt.title('Loss plot')
plt.xlabel('epochs')
plt.ylabel('Loss status')
plt.show()

plt.scatter(train_x, train_y)
plt.plot(train_x, model.test(train_x), '-r')
plt.show()