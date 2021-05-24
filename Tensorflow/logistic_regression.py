import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class logistic_regression():
    def __init__(self):
        self.epochs = 200
        self.learning_rate = 0.1
        self.w = tf.Variable(tf.random.normal([2, 1], dtype=tf.double), name='weight')
        self.b = tf.Variable(tf.zeros([1], dtype=tf.double), name='bias')
    
    def train_batch(self, x, y):
        with tf.GradientTape() as tape:
            hypothesis = tf.sigmoid( tf.matmul(x, self.w) + self.b )
            eps = 1e-10
            loss = -tf.reduce_mean(y*tf.math.log(hypothesis+eps) + (1-y)*tf.math.log(1-hypothesis+eps))
            loss_w, loss_b = tape.gradient(loss, [self.w, self.b])
        self.w.assign_sub(self.learning_rate * loss_w)
        self.b.assign_sub(self.learning_rate * loss_b)
        return loss

    def train(self, x, y):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=16).batch(8)
        loss_mem = []

        for e in range(self.epochs):
            for (x, y) in dataset:
                loss = self.train_batch(x, y)
            loss_mem.append(loss)
        return loss_mem

    def predict(self, x):
        return tf.sigmoid( tf.matmul(x, self.w) + self.b )
    
    def eval(self, x, y):
        hypothesis = tf.sigmoid( tf.matmul(x, self.w) + self.b )
        y_hat = np.round(hypothesis, 0)
        accuracy = np.sum(y_hat == y) / len(y)
        return accuracy


train_x = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [3, 2], [2, 3], [4, 1], 
           [5, 2], [5, 3], [6, 2], [6, 4], [7, 1], [1, 7], [2, 5], [3, 5]]
train_y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
test_x = [[2, 3], [3, 1], [4, 0], [6, 3], [1, 10], [5, 5]]
test_y = [0, 0, 0, 1, 1, 1]


train_x = np.array(train_x, dtype=np.double)
train_y = np.array(train_y, dtype=np.double)
test_x = np.array(test_x, dtype=np.double)
test_y = np.array(test_y, dtype=np.double)
train_y = np.expand_dims(train_y, axis=1)
test_y = np.expand_dims(test_y, axis=1)


plt.scatter(train_x[:,0], train_x[:,1], c=train_y)
plt.show()

model = logistic_regression()
loss_mem = model.train(train_x, train_y)

epochs_x = list(range(len(loss_mem)))
plt.plot(epochs_x, loss_mem)
plt.show()

print( model.eval(test_x, test_y) )