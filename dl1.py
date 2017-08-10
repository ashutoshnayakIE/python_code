import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x_vals = np.load("X2.npy")
y_vals = np.load("Y2.npy")

corr = np.load("correlation.npy")

Xpred = x_vals
x_vals = np.subtract(x_vals, y_vals)

submit = pd.read_csv("submit.csv")
products = pd.read_csv("timeStamp.csv")['product_id'].tolist()
users = submit['user_id'].tolist()
count = submit['average'].tolist()

from tensorflow.python.framework import ops

ops.reset_default_graph()
import tensorflow as tf

p = len(products)

seed = 0
tf.set_random_seed(seed)
np.random.seed(seed)

x_data = tf.placeholder(shape=[None, len(products)], dtype=tf.float32)
y_data = tf.placeholder(shape=[None, len(products)], dtype=tf.float32)

learningRate = tf.placeholder(dtype=tf.float32)

train_indices = np.random.choice(len(x_vals), int(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

corr = tf.cast(corr, tf.float32)
m = 6
A = tf.Variable(tf.random_uniform(shape=[p, p], maxval=tf.sqrt(m / (p + 0.0)), minval=-tf.sqrt(m / (p + 0.0))))

output1 = tf.add(corr, A)
output4 = tf.matmul(x_data, output1)

loss = tf.reduce_sum(tf.squared_difference(output4, y_data))  # + 0.01*tf.reduce_sum(tf.abs(A))
my_opt = tf.train.AdamOptimizer(learningRate)

my_opt = tf.train.AdamOptimizer(learningRate)
gradients, variables = zip(*my_opt.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimize = my_opt.apply_gradients(zip(gradients, variables))
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()

saver = tf.train.Saver()
# sess.run(init)
saver.restore(sess, "/home/ashutoshnayak1991/model.ckpt")
np.save("/home/ashutoshnayak1991/A.npy",sess.run(A))
train_loss = []
test_loss = []

iterations = 2000

epoch = 1
r_rate = 0.001

for i in range(iterations):
    batch_size = 2
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)

    rand_x = x_vals_train[rand_index]
    rand_y = y_vals_train[rand_index]

    if i % 200 == 0:
        epoch += 1
        rate = r_rate / epoch

    sess.run(train_step, feed_dict={x_data: rand_x, y_data: rand_y, learningRate: rate})

    train_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_train, y_data: y_vals_train})
    train_loss.append(train_temp_loss)

    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_data: y_vals_test})
    test_loss.append(test_temp_loss)

    print (i, train_temp_loss, test_temp_loss)

    if (i + 1) == iterations:
        save_path = saver.save(sess, "/home/ashutoshnayak1991/model.ckpt")
    '''
    if (i+1) == iterations:
        submitTry = pd.read_csv("best.csv")
        output1 = tf.add(corr,A)
	output4 = tf.matmul(x_data,output1)
        final = sess.run(output4, feed_dict={x_data: Xpred})
        for t in range(15000):
            countss = int(count[t])
            s = ""
            predicted_products = np.argsort(final[t])[::-1]
            for u in range(countss):
                s += str(products[predicted_products[u]])
                if u < countss - 1:
                    s += " "
            submit.iloc[t+15000, 1] = s


        submit.to_csv("/home/ashutoshnayak1991/submitTry.csv")
    '''