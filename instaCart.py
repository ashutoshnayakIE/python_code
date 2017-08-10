import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path for spark source folder
'''
os.environ['SPARK_HOME']="/home/ashutosh/Dropbox/spark-2.1.1-bin-hadoop2.7"

# Append pyspark  to Python Path
sys.path.append("/home/ashutosh/Dropbox/spark-2.1.1-bin-hadoop2.7/python")
sys.path.append("/home/ashutosh/Dropbox/spark-2.1.1-bin-hadoop2.7/python/lib/py4j-0.10.4-src.zip")


try:
    from pyspark import SparkContext
    from pyspark import SparkConf

except ImportError as e:
    sys.exit(1)

conf = SparkConf().setMaster("local[*]").setAppName("My App")
sc = SparkContext(conf=conf)


order_data = pd.read_csv("/home/ashutosh/Desktop/instacart/orders.csv")
order_data = order_data.sort_values(by=['user_id', 'order_id'], ascending=[1, 1])
order_data.to_csv("/home/ashutosh/Desktop/instacart/orders.csv")
'''

# uploading the dataFiles----------------------------------------------------------------------------
'''
# isles file contains  aisle_id,aisle
aisles = pd.read_csv("/home/ashutosh/Desktop/instacart/aisles.csv")
aisle_dict = aisles.set_index('aisle_id')['aisle'].to_dict()

# departments has department_id,department
departments = pd.read_csv("/home/ashutosh/Desktop/instacart/departments.csv")
departments_dict = departments.set_index('department_id')['department'].to_dict()

# products has  product_id,product_name,aisle_id,department_id
products = pd.read_csv("/home/ashutosh/Desktop/instacart/products.csv")

# order_products_* has order_id,product_id,add_to_cart_order,reordered

# converting the orders in order.csv in sorted_orders according to the order_id

order_data = pd.read_csv("/home/ashutosh/Desktop/instacart/orders.csv")
order_data = order_data.sort_values(by=['order_id', 'user_id'], ascending=[1, 1])
test_data = order_data[order_data['eval_set']=="test"]
sorted_orders = pd.DataFrame(order_data)
sorted_orders.to_csv('/home/ashutosh/Desktop/instacart/Ordered_orders.csv')

# adding a combined data with specific users to learn the data...
# created the working data for 1000 test case users

# selecting different datasets based on 100 users and...
# rest of the analysis is based on these 100 users to identify the pattern

# saving the users

order_data = pd.read_csv("/home/ashutosh/Desktop/instacart/Ordered_orders.csv")
#order_data = order_data.drop(order_data.columns[0],axis=1)
#test_data = order_data[order_data['eval_set']=="test"]
#del test_data['eval_set']
#np.savetxt("/home/ashutosh/Desktop/instacart/users.csv", test_data['user_id'].unique()[0:1000], delimiter=",")

#users = test_data['user_id'].unique()
#orders = pd.Series(order_data[order_data['user_id'].isin(users)]['order_id'])

train_data = pd.read_csv("/home/ashutosh/Desktop/instacart/order_products__train.csv")
del train_data['add_to_cart_order']

prior_data = pd.read_csv("/home/ashutosh/Desktop/instacart/order_products__prior.csv")
del prior_data['add_to_cart_order']

#train_data = train_data[train_data['order_id'].isin(orders)]
#prior_data = prior_data[prior_data['order_id'].isin(orders)]

train_data = train_data.merge(order_data[['order_id','user_id']],left_on='order_id',right_on='order_id',how='inner')
prior_data = prior_data.merge(order_data[['order_id','user_id']],left_on='order_id',right_on='order_id',how='inner')

concatenate_frames = [prior_data,train_data]
working_data = pd.concat(concatenate_frames)
# sorting the working _data by the user_id
working_data = working_data.merge(products[['product_id','aisle_id','department_id']],left_on='product_id',right_on='product_id',how='inner')
working_data = working_data.sort_values(by=['user_id', 'order_id'], ascending=[1, 1])

print working_data.head()
working_data.to_csv('/home/ashutosh/Desktop/instacart/Working_orders.csv')
'''
# ------------------------------------------------------------------------------
# ---------------------------- recent work -------------------------------------

# 1) updating the dataset for username, aisle id and department id
'''
products = pd.read_csv("/home/ashutosh/Desktop/instacart/products.csv")
orders = pd.read_csv("/home/ashutosh/Desktop/instacart/orders.csv")

prior_data = pd.read_csv("/home/ashutosh/Desktop/instacart/order_products__prior.csv")
del prior_data['add_to_cart_order']

train_data = pd.read_csv("/home/ashutosh/Desktop/instacart/order_products__train.csv")
del train_data['add_to_cart_order']

prior_data = prior_data.merge(products[['product_id','aisle_id','department_id']],left_on='product_id',right_on='product_id',how='inner')
prior_data = prior_data.merge(orders[['order_id','user_id']],left_on='order_id',right_on='order_id',how='inner')
prior_data = prior_data.sort_values(by=['user_id', 'order_id'], ascending=[1, 1])
prior_data.to_csv('/home/ashutosh/Desktop/instacart/prior_data.csv')

train_data = train_data.merge(products[['product_id','aisle_id','department_id']],left_on='product_id',right_on='product_id',how='inner')
train_data = train_data.merge(orders[['order_id','user_id']],left_on='order_id',right_on='order_id',how='inner')
train_data = train_data.sort_values(by=['user_id', 'order_id'], ascending=[1, 1])
train_data.to_csv('/home/ashutosh/Desktop/instacart/train_data.csv')
'''

# 2) finding the number of times a particular product has been ordered
# it will help in reducing the dimension we have to study

# finding the number of times a product has been ordered
# and also finding how many times it has been reordered
'''
prior_data = sc.textFile("/home/ashutosh/Desktop/instacart/prior_data.csv")
train_data = sc.textFile("/home/ashutosh/Desktop/instacart/train_data.csv")

def parse(x):
    x = x.split(",")
    return (int(x[2]),(1,int(x[3])))

prior_data = prior_data.filter(lambda x: "order_id" not in x).map(parse)
prior_data = prior_data.reduceByKey(lambda x,y:(x[0]+y[0],x[1]+y[1])).map(lambda x:(x[0],x[1][0],x[1][1]))

countP = prior_data.collect()

train_data = train_data.filter(lambda x: "order_id" not in x).map(parse)
train_data = train_data.reduceByKey(lambda x,y:(x[0]+y[0],x[1]+y[1])).map(lambda x:(x[0],x[1][0],x[1][1]))

countT = train_data.collect()

label = ['product','count','reorder']
countP = pd.DataFrame.from_records(countP, columns=label)
countT = pd.DataFrame.from_records(countT, columns=label)

countP.index = countP['product']
countT.index = countT['product']
del countP['product']
del countT['product']
# adding two dataframes on equal indices
count = countP.add(countT,fill_value=0)
count.to_csv("/home/ashutosh/Desktop/instacart/countOrders.csv")

'''
# 3) finding the average number of orders by each consumer
'''
prior_data = sc.textFile("/home/ashutosh/Desktop/instacart/prior_data.csv")
train_data = sc.textFile("/home/ashutosh/Desktop/instacart/train_data.csv")

def parse(x):
    x = x.split(",")
    return ((int(x[6]),int(x[1])),1)

prior_data = prior_data.filter(lambda x: "order_id" not in x).map(parse)
average_prior_data = prior_data.reduceByKey(lambda x,y: x+y).map(lambda x:(x[0][0],x[1])).cache()
sumP = average_prior_data.reduceByKey(lambda x,y: x+y)
countP = average_prior_data.map(lambda x:(x[0],1)).reduceByKey(lambda x,y: x+y)
sum_prior = sumP.collect()
count_prior = countP.collect()

train_data = train_data.filter(lambda x: "order_id" not in x).map(parse)
average_train_data = train_data.reduceByKey(lambda x,y: x+y).map(lambda x:(x[0][0],x[1])).cache()
sumT = average_train_data.reduceByKey(lambda x,y: x+y)
countT = average_train_data.map(lambda x:(x[0],1)).reduceByKey(lambda x,y: x+y)
sum_train = sumT.collect()
count_train = countT.collect()

label = ['user_id','sum']
sumP = pd.DataFrame.from_records(sum_prior, columns=label)
sumT = pd.DataFrame.from_records(sum_train, columns=label)
sumP.index = sumP['user_id']
sumT.index = sumT['user_id']

label = ['user_id','count']
countP = pd.DataFrame.from_records(count_prior, columns=label)
countT = pd.DataFrame.from_records(count_train, columns=label)
countP.index = countP['user_id']
countT.index = countT['user_id']

# adding two dataframes on equal indices
sumAll = sumP.add(sumT,fill_value=0)
count = countP.add(countT,fill_value=0)

averageOrders = sumAll.merge(count[['user_id','count']],left_on='user_id',right_on='user_id',how='inner')
averageOrders["average"] = averageOrders["sum"]/averageOrders["count"]
averageOrders["user_id"] = averageOrders["user_id"]/2 # becasue adding dataframes add all the columns

averageOrders.to_csv("/home/ashutosh/Desktop/instacart/averageOrders.csv")
'''
# plotting the graphs to find out which products have to be considered
# 4) using that products for developing the training set
'''
global products

count = pd.read_csv("/home/ashutosh/Desktop/instacart/countOrders.csv")
count = count[count["count"] > 50000]
products = count['product'].tolist()

# converting the input file into the usable format

prior_data = sc.textFile("/home/ashutosh/Desktop/instacart/prior_data.csv")
train_data = sc.textFile("/home/ashutosh/Desktop/instacart/train_data.csv")

def parse(x):
    x = x.split(",")
    y = np.array([0]*len(products))
    y[products.index(int(x[2]))] += 1
    return (int(x[6]),y)

prior_data = prior_data.filter(lambda x: "order_id" not in x and int(x.split(",")[2]) in products).map(parse).reduceByKey(lambda x,y: x+y)
prior_data_train = prior_data.collect()

train_data = train_data.filter(lambda x: "order_id" not in x and int(x.split(",")[2]) in products).map(parse).reduceByKey(lambda x,y: x+y)
train_data_train = train_data.collect()

label = ['user_id','input']
PD = pd.DataFrame.from_records(prior_data_train, columns=label)
label = ['user_id','output']
TD = pd.DataFrame.from_records(train_data_train, columns=label)

# merges the two dataframes for input and output

data = PD.merge(TD[['user_id','output']],left_on='user_id',right_on='user_id',how='inner')
data["zeros"] = 0
data = data.sort_values(by=['user_id', 'zeros'], ascending=[1, 1])
del data['zeros']
PD["zeros"] = 0
PD = PD.sort_values(by=['user_id', 'zeros'], ascending=[1, 1])
del PD['zeros']

data.to_csv("/home/ashutosh/Desktop/instacart/data_in_format.csv")
PD.to_csv("/home/ashutosh/Desktop/instacart/prior_in_format.csv")
'''
# 5) developing the machine learning algorithm to find the next products
'''
count = pd.read_csv("/home/ashutosh/Desktop/instacart/countOrders.csv")
count = count[count["count"] > 50000]
products = count['product'].tolist()
p = len(products)

data = pd.read_csv("/home/ashutosh/Desktop/instacart/data_in_format.csv")
prior_in_format = pd.read_csv("/home/ashutosh/Desktop/instacart/prior_in_format.csv")

def formatting(x):
    newstr = x.replace("[ ", "[")
    newstr = newstr.replace("[", "")
    newstr = newstr.replace("]", "")
    newstr = newstr.replace("\n", "")
    newstr = newstr.replace("  ", " ")
    x = newstr.split(" ")
    return [int(x[i]) for i in range(len(products))]

x_val = data['input'].apply(lambda x:formatting(x))
y_val = data['output'].apply(lambda x:formatting(x))
prior_val = prior_in_format['input'].apply(lambda x:formatting(x))

x_vals = np.zeros((len(data),p))
y_vals = np.zeros((len(data),p))
prior_vals = np.zeros((len(prior_in_format),p))

for rows in range(len(data)):
    for cols in range(p):
        x_vals[rows][cols] = x_val[rows][cols]
        y_vals[rows][cols] = y_val[rows][cols]

for rows in range(len(prior_in_format)):
    for cols in range(p):
        prior_vals[rows][cols] = prior_val[rows][cols]

# saving the data in the usable format

np.save("/home/ashutosh/Desktop/instacart/x_vals.npy",x_vals)
np.save("/home/ashutosh/Desktop/instacart/y_vals.npy",y_vals)
np.save("/home/ashutosh/Desktop/instacart/prior_vals.npy",prior_vals)

'''
# 6) collecting the order_id, user_id, average orders for all the consumers with test statistics
'''
order_data = pd.read_csv("/home/ashutosh/Desktop/instacart/orders.csv")
test_data = order_data[order_data['eval_set']=="test"]
test_data = test_data[["order_id","user_id"]]

average_orders = pd.read_csv("/home/ashutosh/Desktop/instacart/averageOrders.csv")
test_data = test_data.merge(average_orders[['user_id','average']],left_on='user_id',right_on='user_id',how='inner')
test_data.to_csv("/home/ashutosh/Desktop/instacart/test_data.csv")
'''

# 7) machine learning algorithm
ppp = np.load("/home/ashutosh/Desktop/instacart/huha.npy")
print ppp.shape
from tensorflow.python.framework import ops
ops.reset_default_graph()
import tensorflow as tf
sess = tf.Session()
h1 = 80
m = 6
p = 149
A1 = tf.Variable(tf.random_uniform(shape=[p,h1],maxval=tf.sqrt(m/(p+h1+0.0)),minval=-tf.sqrt(m/(p+h1+0.0))))
b1 = tf.Variable(tf.constant([0.0]*h1))
init = tf.global_variables_initializer()
sess.run(init)
print sess.run(A1)
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
#sess.run(init)
saver.restore(sess, "/home/ashutosh/Desktop/instacart/model_non_lstm_relu.ckpt")
print " ----------------------------- "
np.save("/home/ashutosh/Desktop/instacart/huha.npy",sess.run(A1))


x_vals = np.load("/home/ashutosh/Desktop/instacart/149/RFX.npy")
y_vals = np.load("/home/ashutosh/Desktop/instacart/149/RFY.npy")

from tensorflow.python.framework import ops
ops.reset_default_graph()
import tensorflow as tf


data_in_format = pd.read_csv("/home/ashutosh/Desktop/instacart/149/RFUserInfo.csv")
users = data_in_format['user_id'].tolist()
averageOrders = data_in_format['average'].tolist()


productCO = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/CO.csv")
products = productCO['product_id'].tolist()[0:3500]
p = len(products)
dist = productCO['count'].tolist()
sum_dist = sum(dist)
dist = [dist[d]/(1*float(sum_dist)) for d in range(p)]

seed = 100
tf.set_random_seed(seed)
np.random.seed(seed)
batch_size = 256

x_data = tf.placeholder(shape=[None, len(products)], dtype=tf.float32)
y_data = tf.placeholder(shape=[None, len(products)], dtype=tf.float32)

learningRate = tf.placeholder(dtype=tf.float32)

train_indices = np.random.choice(len(x_vals), int(len(x_vals)*0.8), replace=False)
test_indices  = np.array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test  = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test  = y_vals[test_indices]

# setting the variables for deep learning
# all 60 was the best
h1 = 80
h2 = 80
h3 = 80
h4 = 80
h5 = 20

m = 6
# three layers perform the best
A1 = tf.Variable(tf.random_uniform(shape=[p,h1],maxval=tf.sqrt(m/(p+h1+0.0)),minval=-tf.sqrt(m/(p+h1+0.0))))
b1 = tf.Variable(tf.constant([0.0]*h1))
A2 = tf.Variable(tf.random_uniform(shape=[h1,h2],maxval=tf.sqrt(m/(h1+h2+0.0)),minval=-tf.sqrt(m/(h1+h2+0.0))))
b2 = tf.Variable(tf.constant([0.0]*h2))
A3 = tf.Variable(tf.random_uniform(shape=[h2,h3],maxval=tf.sqrt(m/(h2+p+0.0)),minval=-tf.sqrt(m/(h2+p+0.0)))) #tf.Variable(tf.random_normal(shape=[h2,h3],stddev= 2))
b3 = tf.Variable(tf.constant([0.0]*h3))
A4 = tf.Variable(tf.random_uniform(shape=[h3,h4],maxval=tf.sqrt(m/(h3+h4+0.0)),minval=-tf.sqrt(m/(h3+h4+0.0)))) #tf.Variable(tf.random_normal(shape=[h2,h3],stddev= 2))
b4 = tf.Variable(tf.constant([0.0]*h4))
A5 = tf.Variable(tf.random_uniform(shape=[h4,h5],maxval=tf.sqrt(m/(h4+h5+0.0)),minval=-tf.sqrt(m/(h4+h5+0.0)))) #tf.Variable(tf.random_normal(shape=[h2,h3],stddev= 2))
b5 = tf.Variable(tf.constant([0.0]*h5))
A6 = tf.Variable(tf.random_uniform(shape=[h5,p],maxval=tf.sqrt(m/(h5+p+0.0)),minval=-tf.sqrt(m/(h5+p+0.0)))) #tf.Variable(tf.random_normal(shape=[h2,h3],stddev= 2))
b6 = tf.Variable(tf.constant([0.0]*p))

output1 = tf.nn.relu6(tf.add(tf.matmul(x_data, A1),b1))
output2 = tf.nn.relu6(tf.add(tf.matmul(output1, A2),b2))
output3 = tf.nn.relu6(tf.add(tf.matmul(output2, A3),b3))
output4 = tf.nn.relu6(tf.add(tf.matmul(output3, A4),b4))
output5 = tf.nn.relu6(tf.add(tf.matmul(output4, A5),b5))
output6 = tf.nn.sigmoid(tf.add(tf.matmul(output5, A6),b6))

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=output3))#+0.01*(tf.reduce_sum(tf.abs(A1))+tf.reduce_sum(tf.abs(A2))+tf.reduce_sum(tf.abs(A3)))#+tf.reduce_sum(tf.abs(A4))+tf.reduce_sum(tf.abs(A5))+tf.reduce_sum(tf.abs(A6))+tf.reduce_sum(tf.abs(b1))+tf.reduce_sum(tf.abs(b2))+tf.reduce_sum(tf.abs(b3))+tf.reduce_sum(tf.abs(b4))+tf.reduce_sum(tf.abs(b5))+tf.reduce_sum(tf.abs(b6)))
#loss = tf.reduce_mean(tf.squared_difference(output5, y_data))#-0.1*(tf.reduce_mean(tf.abs(output4)))+0.001*(tf.reduce_sum(tf.abs(A1))+tf.reduce_sum(tf.abs(A2))+tf.reduce_sum(tf.abs(A3)))
#loss = tf.reduce_mean(tf.abs(tf.subtract(output4, y_data)))#-0.1*(tf.reduce_mean(tf.abs(output4)))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, logits=output6))
#loss = -tf.reduce_mean((y_data * tf.sigmoid(output4)) + ((1 - y_data) * tf.sigmoid(1 - output4)))
#loss = tf.reduce_mean(tf.losses.hinge_loss(logits=output5, labels=y_data))


TP = tf.reduce_sum(output6*y_data)
FP = tf.reduce_sum(output6*(1-y_data))
FN = tf.reduce_sum((1-output6)*y_data)
precision = TP/(TP+FP+0.0)
recall = TP/(TP+FN+0.0)
loss = -2000*precision*recall
my_opt = tf.train.AdamOptimizer(learningRate)

d1 = 0.2
d2 = 0.2
d3 = 0.2
d4 = 0.2
d5 = 0.2
d6 = 0.2

kp1   = tf.placeholder(dtype=tf.float32)
drop1 = tf.nn.dropout(output1,kp1)
kp2   = tf.placeholder(dtype=tf.float32)
drop2 = tf.nn.dropout(output2,kp2)
kp3   = tf.placeholder(dtype=tf.float32)
drop3 = tf.nn.dropout(output3,kp3)
kp4   = tf.placeholder(dtype=tf.float32)
drop4 = tf.nn.dropout(output4,kp4)
kp5   = tf.placeholder(dtype=tf.float32)
drop5 = tf.nn.dropout(output5,kp5)
kp6   = tf.placeholder(dtype=tf.float32)
drop6 = tf.nn.dropout(output6,kp6)

# points about the optimization method used
# 6,10: hidden layers
# GradientDescentOptimizer(0.0001) is going down but very slow
# GradientDescentOptimizer(0.001) is performing better than above
# MomentumOptimizer(0.01,0.96,True) is much faster than above: reaches best result so far, batchsize 5000
# AdamOptimizer(0.01) is also fast and personal fav of the author, works well with smaller rate
# yoshua: keep on adding layers until error decreases
# 5 hidden layers not behaving well (becoming saturated); no randomness for biases, but do as default
# initialization: A4 = tf.Variable(tf.random_uniform(shape=[h3,p],maxval=tf.sqrt(6/(2.0*p)),minval=-tf.sqrt(6/(2.0*p))))
# b4 = tf.Variable(tf.random_normal(shape=[1,p])) has been replaced based on the book
# AdamOptimizer(0.005) best value so far, for 3 layers (relu6), batch size 2500

train_step = my_opt.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
#sess.run(init)
saver.restore(sess, "/home/ashutosh/Desktop/instacart/model_non_lstm_relu.ckpt")

print sess.run(A1)

# Initialize the loss vectors
loss_vec = []
test_loss = []

iterations = 30000

epoch = 1
r_rate = 0.00005
for i in range(iterations):
    batch_size = 64#int(np.random.uniform(16,32,1))
    # Choose random indices for batch selection
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    # Get random batch
    rand_x = x_vals_train[rand_index]
    rand_y = y_vals_train[rand_index]

    if i%1000 == 0:
        epoch += 0.05
        rate = r_rate/epoch

    # Run the training step
    sess.run(train_step, feed_dict={x_data: rand_x, y_data:rand_y, learningRate: rate, kp1:d1, kp2:d2, kp3:d3, kp4:d4})

    # Get and store the train loss
    temp_loss = sess.run(loss, feed_dict={x_data: x_vals_train, y_data: y_vals_train})
    loss_vec.append(temp_loss)
    # Get and store the test loss
    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_data: y_vals_test})
    test_loss.append(test_temp_loss)
    #print (i+1, temp_loss,test_temp_loss)

    if (i+1)%3000 == 0:

        output1 = tf.nn.relu6(tf.add(tf.matmul(x_data, A1), b1))
        output2 = tf.nn.relu6(tf.add(tf.matmul(output1, A2), b2))
        output3 = tf.nn.relu6(tf.add(tf.matmul(output2, A3), b3))
        output4 = tf.nn.relu6(tf.add(tf.matmul(output3, A4), b4))
        output5 = tf.nn.relu6(tf.add(tf.matmul(output4, A5), b5))
        output6 = tf.nn.sigmoid(tf.add(tf.matmul(output5, A6), b6))

        final = sess.run(output6,feed_dict={x_data: x_vals})
        tp = 0
        fp = 0
        fn = 0
        for t in range(len(x_vals)):
            countss = int(0.4*averageOrders[t])
            predicted_products = np.argsort(final[t])[::-1]
            for u in range(countss):
                if y_vals[t][predicted_products[u]] >= 1:
                    tp += 1
                else:
                    fp += 1
            for u in range(countss, len(y_vals[t])):
                if y_vals[t][predicted_products[u]] >= 1:
                    fn += 1

        pre = tp / (tp + fp + 0.0)
        rec = tp / (tp + fn + 0.0)
        print ("output from deep learning",tp,fp,fn, 2 * pre * rec / (pre + rec))


    if (i+1) == iterations:
        save_path = saver.save(sess, "/home/ashutosh/Desktop/instacart/model_non_lstm_relu.ckpt")

plt.plot(loss_vec, 'k-',label='Train Loss')
plt.plot(test_loss, 'r--',label='Test Loss')
plt.title('Loss per Generation')

plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
