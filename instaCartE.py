# here is the list of data I am making for different learning schemes
'''

1. x_vals,y_vals : randomForest learning with 63 products
2. x_valsN, y_valsN : randomForest learning with 149 products
3. x_valsRNN, y_valsRNN : lstm learning with 149 products

4. x_lstm : testing data for lstm_users with 149 products
5. x_non_lstm : testing data for non_lstm_users with 63 products

6. finalUsers is the user information for training dataset of non_lstm with 149 products
7. data_in_format['user_id'] contains the information on dataset for x_vals and y_vals for 63 products

7. testUsers is the user information for the testing data for lstm
8. non_lstm_testUsers is the user information for testing data for non lstm users with 63 products

'''

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# initializing spark
'''
# Path for spark source folder
os.environ['SPARK_HOME']="/home/ashutosh/Dropbox/spark-2.1.1-bin-hadoop2.7"
os.environ['PYSPARK_SUBMIT_ARGS'] = ' --driver-memory 50g pyspark-shell'

# Append pyspark  to Python Path
sys.path.append("/home/ashutosh/Dropbox/spark-2.1.1-bin-hadoop2.7/python")
sys.path.append("/home/ashutosh/Dropbox/spark-2.1.1-bin-hadoop2.7/python/lib/py4j-0.10.4-src.zip")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

except ImportError as e:
    sys.exit(1)

conf = SparkConf().setMaster("local[*]").setAppName("My App")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
'''
# converting the data into the format of 149 products
# ====================================================================================
global products,p

count = pd.read_csv("/home/ashutosh/Desktop/instacart/countOrders.csv")
count = count[count["count"] > 25000]
products = count['product'].tolist()
p = len(products)
'''
working_data = sc.textFile("/home/ashutosh/Desktop/instacart/working_data.csv")

def parse1(x):
    x = x.split(",")
    y = np.array([0]*len(products))
    if int(x[2]) in products:
        y[products.index(int(x[2]))] += 1
    return ((int(x[3]),int(x[4])),y) # user_id, order_number, y

def parse2(x):
    return (x[0][1],(x[0][0],x[1].tolist())) # order_number, user_id, y

def parse3(x):
    return (x[1][0],x[1][1])

working_data = working_data.filter(lambda x: "order_id" not in x).map(parse1).reduceByKey(lambda  x,y: x+y).map(parse2).sortByKey().map(parse3).sortByKey()
working_data = working_data.reduceByKey(lambda x,y:x+[y]).sortByKey()

working_data_format = working_data.collect()
label = ['user_id','input']
working_data_format = pd.DataFrame.from_records(working_data_format,columns=label)

average_orders = pd.read_csv("/home/ashutosh/Desktop/instacart/averageOrders.csv")
working_data_format = working_data_format.merge(average_orders[['user_id','orders']],left_on='user_id',right_on='user_id',how='inner')

working_data_format.to_csv("/home/ashutosh/Desktop/instacart/149/working_data_format.csv")
'''

# converting it into random forest framework
'''
RFdata = pd.read_csv("/home/ashutosh/Desktop/instacart/149/working_data_format.csv")
average_orders = pd.read_csv("/home/ashutosh/Desktop/instacart/averageOrders.csv")
RFdata = RFdata.merge(average_orders[['user_id','average']],left_on='user_id',right_on='user_id',how='inner')

def formattingO(x):
    x = str(x)
    x = x.replace("[","")
    x = x.replace("]", "")
    x = x.split(",")
    y = []
    for i in range(p):
        y.insert(0,int(x[-1-i]))
    return y

def formattingI(x):
    x = str(x)
    x = x.replace("[","")
    x = x.replace("]", "")
    x = x.split(",")
    y = [0]*p
    for i in range(len(x)/p-1):
        for j in range(p):
            y[j] += int(x[i*p+j])
    return y

RFdata['response'] = RFdata['input'].apply(lambda x: formattingO(x))
RFdata['predictor'] = RFdata['input'].apply(lambda x: formattingI(x))

x_val = RFdata['predictor']
y_val = RFdata['response']

RFX = np.zeros((len(RFdata),p))
RFY = np.zeros((len(RFdata),p))

for rows in range(len(RFdata)):
    for cols in range(p):
        RFY[rows][cols] = y_val[rows][cols]
        RFX[rows][cols] = x_val[rows][cols]

np.save("/home/ashutosh/Desktop/instacart/149/RFX.npy",RFX)
np.save("/home/ashutosh/Desktop/instacart/149/RFY.npy",RFY)

RFUserInfo = RFdata[['user_id','orders','average']]
RFUserInfo.to_csv("/home/ashutosh/Desktop/instacart/149/RFUserInfo.csv")
'''

# generating the data for lstm users
'''
working_data = pd.read_csv("/home/ashutosh/Desktop/instacart/149/working_data_format.csv")
working_data = working_data[['user_id','input','orders','average']]
working_data = working_data[working_data['orders']>11]

def formatting2(x):
    x = str(x)
    x = x.replace("[","")
    x = x.replace("]", "")
    x = x.split(",")
    a = []
    for i in range(11):
        b = []
        for j in range(len(products)):
            b.insert(0,int(x[-1-j-len(products)*i]))
        a.insert(0,b)
    return a
working_data['input'] = working_data['input'].apply(lambda x: formatting2(x))
working_data['output'] = working_data['input'].apply(lambda x: x[-1])
working_data['input'] = working_data['input'].apply(lambda x: x[0:10])

working_data.to_csv("/home/ashutosh/Desktop/instacart/149/LSTMdata.csv")
'''
# creating the data for lstm
'''
working_data = pd.read_csv("/home/ashutosh/Desktop/instacart/149/LSTMdata.csv")

x_val = working_data['input']
y_val = working_data['output']

x_vals_rnn = np.zeros((len(working_data),10,p))
y_vals_rnn = np.zeros((len(working_data),p))

for rows in range(len(working_data)):
    y = y_val[rows]
    y = y.replace("[","")
    y = y.replace("]", "")
    y = y.replace(" ", "")
    y = y.replace("'", "")
    y = y.split(",")
    for cols in range(p):
        y_vals_rnn[rows][cols] = int(y[cols])
    x = x_val[rows]
    x = x.replace("[","")
    x = x.replace("]", "")
    x = x.replace(" ", "")
    x = x.replace("'", "")
    x = x.split(",")
    for cols1 in range(10):
        for cols2 in range(p):
            x_vals_rnn[rows][cols1][cols2] = int(x[cols1*p+cols2])

np.save("/home/ashutosh/Desktop/instacart/149/LSTMX.npy",x_vals_rnn)
np.save("/home/ashutosh/Desktop/instacart/149/LSTMY.npy",y_vals_rnn)

LSTMUserInfo = working_data[['user_id','orders','average']]
LSTMUserInfo.to_csv("/home/ashutosh/Desktop/instacart/149/LSTMUserInfo.csv")
'''
# generating the X values for the lstm and RF
'''
submit = pd.read_csv("/home/ashutosh/Desktop/instacart/149/submit.csv")
predUsers = submit['user_id'].tolist()

LSTMUserInfo = pd.read_csv("/home/ashutosh/Desktop/instacart/149/LSTMUserInfo.csv")
LSTMUserInfo = LSTMUserInfo[LSTMUserInfo['user_id'].isin(predUsers)]
LSTMUsers = LSTMUserInfo['user_id'].tolist()
LSTMindex = list(set(predUsers)&set(LSTMUsers))
LSTMindex = [LSTMUsers.index(i) for i in LSTMindex]

X = np.load("/home/ashutosh/Desktop/instacart/149/LSTMX.npy")
Y = np.load("/home/ashutosh/Desktop/instacart/149/LSTMY.npy")

testLSTMX = X[LSTMindex]
testLSTMY = Y[LSTMindex]

test = np.zeros((len(testLSTMX),10,p))
for i in range(len(testLSTMX)):
    for j in range(10):
        for k in range(p):
            if j < 9:
                test[i][j][k] = testLSTMX[i][j+1][k]
            else:
                test[i][j][k] = testLSTMY[i][k]

np.save("/home/ashutosh/Desktop/instacart/149/testLSTMX.npy",test)
LSTMUserInfo.to_csv("/home/ashutosh/Desktop/instacart/149/LSTMUserInfoTest.csv")

RFUserInfo = pd.read_csv("/home/ashutosh/Desktop/instacart/149/RFUserInfo.csv")
RFUserInfo = RFUserInfo[RFUserInfo['user_id'].isin(predUsers)]
RFUsers = RFUserInfo['user_id'].tolist()
RFindex = list(set(predUsers)&set(RFUsers))
RFindex = [RFUsers.index(i) for i in RFindex]

X = np.load("/home/ashutosh/Desktop/instacart/149/RFX.npy")
Y = np.load("/home/ashutosh/Desktop/instacart/149/RFY.npy")

testRFX = np.add(X[RFindex],Y[RFindex])
print testRFX.shape
np.save("/home/ashutosh/Desktop/instacart/149/testRFX.npy",testRFX)
RFUserInfo.to_csv("/home/ashutosh/Desktop/instacart/149/RFUserInfoTest.csv")
'''

# developing a smaller training set for random forest
'''
userInfo = pd.read_csv("/home/ashutosh/Desktop/instacart/149/RFUserInfo.csv")
allUsers = userInfo['user_id'].tolist()
userInfo = userInfo[userInfo['orders']>10]
users = userInfo['user_id'].tolist()
usersIndex = [allUsers.index(i) for i in users]

X = np.load("/home/ashutosh/Desktop/instacart/149/RFX.npy")
Y = np.load("/home/ashutosh/Desktop/instacart/149/RFY.npy")

RFXs = X[usersIndex]
RFYs = Y[usersIndex]

np.save("/home/ashutosh/Desktop/instacart/149/RFXs.npy",RFXs)
np.save("/home/ashutosh/Desktop/instacart/149/RFYs.npy",RFYs)
userInfo.to_csv("/home/ashutosh/Desktop/instacart/149/RFUserInfos.csv")

print RFXs.shape,RFYs.shape,len(userInfo)
'''
# ====================================================================================

# creating orders data with sorted user id and then order_number
'''
order_data = pd.read_csv("/home/ashutosh/Desktop/instacart/orders.csv")
order_data = order_data.sort_values(by=['user_id', 'order_number'], ascending=[1, 1])
order_data.to_csv("/home/ashutosh/Desktop/instacart/orders.csv")
'''

# refining the prior data and train data
'''
prior_data = pd.read_csv("/home/ashutosh/Desktop/instacart/prior_data.csv")
train_data = pd.read_csv("/home/ashutosh/Desktop/instacart/train_data.csv")
order_data = pd.read_csv("/home/ashutosh/Desktop/instacart/orders.csv")

del prior_data['user_id']
del train_data['user_id']

train_data = train_data.merge(order_data[['order_id','user_id','order_number']],left_on='order_id',right_on='order_id',how='inner')
prior_data = prior_data.merge(order_data[['order_id','user_id','order_number']],left_on='order_id',right_on='order_id',how='inner')

concatenate_frames = [prior_data,train_data]
working_data = pd.concat(concatenate_frames)

working_data = working_data.sort_values(by=['user_id', 'order_number'], ascending=[1, 1])
working_data = working_data[['order_id','product_id','user_id','order_number']]
working_data.to_csv("/home/ashutosh/Desktop/instacart/working_data.csv")
'''
# creating the data
'''
global products,p

count = pd.read_csv("/home/ashutosh/Desktop/instacart/countOrders.csv")
count = count[count["count"] > 25000]
products = count['product'].tolist()
p = len(products)
'''
# converting the input file into the usable format
'''
working_data = sc.textFile("/home/ashutosh/Desktop/instacart/working_data.csv")

def parse1(x):
    x = x.split(",")
    y = np.array([0]*len(products))
    y[products.index(int(x[2]))] += 1
    return ((int(x[3]),int(x[4])),y) # user_id, order_number, y

def parse2(x):
    return (x[0][1],(x[0][0],x[1].tolist())) # order_number, user_id, y

def parse3(x):
    return (x[1][0],x[1][1])

working_data = working_data.filter(lambda x: "order_id" not in x and int(x.split(",")[2]) in products).map(parse1).reduceByKey(lambda  x,y: x+y).map(parse2).sortByKey().map(parse3).sortByKey()
working_data = working_data.reduceByKey(lambda x,y:x+[y]).sortByKey()

working_data_format = working_data.collect()
label = ['user_id','input']
working_data_format = pd.DataFrame.from_records(working_data_format,columns=label)
print working_data_format

average_orders = pd.read_csv("/home/ashutosh/Desktop/instacart/averageOrders.csv")
working_data_format = working_data_format.merge(average_orders[['user_id','orders']],left_on='user_id',right_on='user_id',how='inner')

working_data_format.to_csv("/home/ashutosh/Desktop/instacart/working_data_format.csv")
'''
# converting the dat into a usable form
'''
working_data = pd.read_csv("/home/ashutosh/Desktop/instacart/working_data_format.csv")
working_data = working_data[working_data['orders']>4]

def formatting1(x):
    x = str(x)
    x = x.replace("[","")
    x = x.replace("]", "")
    x = x.split(",")
    return len(x)

working_data['length'] = working_data['input'].apply(lambda x: formatting1(x))
working_data = working_data[working_data['length']>4*p]


def formatting2(x):
    x = str(x)
    x = x.replace("[","")
    x = x.replace("]", "")
    x = x.split(",")
    a = []
    for i in range(5):
        b = []
        for j in range(len(products)):
            b.append(int(x[-1-j-len(products)*i]))
        a.append(b)
    return a
working_data['input'] = working_data['input'].apply(lambda x: formatting2(x))
del working_data['length']
working_data['output'] = working_data['input'].apply(lambda x: x[-1])
working_data['input'] = working_data['input'].apply(lambda x: x[0:4])

average_orders = pd.read_csv("/home/ashutosh/Desktop/instacart/averageOrders.csv")
working_data= working_data.merge(average_orders[['user_id','average']],left_on='user_id',right_on='user_id',how='inner')

finalUsers = working_data[['user_id','orders','average']]
finalUsers.to_csv("/home/ashutosh/Desktop/instacart/finalUsers.csv")
print working_data.head(),len(working_data)

x_val = working_data['input']
y_val = working_data['output']

x_vals_rnn = np.zeros((len(working_data),4,p))
y_vals_rnn = np.zeros((len(working_data),p))

for rows in range(len(working_data)):
    for cols in range(p):
        y_vals_rnn[rows][cols] = y_val[rows][cols]
    for cols1 in range(4):
        for cols2 in range(p):
            x_vals_rnn[rows][cols1][cols2] = x_val[rows][cols1][cols2]

np.save("/home/ashutosh/Desktop/instacart/x_vals_rnn.npy",x_vals_rnn)
np.save("/home/ashutosh/Desktop/instacart/y_vals_rnn.npy",y_vals_rnn)

'''
 # generating the data for test users

# step 1: getting the test data points
'''
order_data = pd.read_csv("/home/ashutosh/Desktop/instacart/orders.csv")
test_users = order_data[order_data['eval_set']=="test"]['user_id'].tolist()

working_data = pd.read_csv("/home/ashutosh/Desktop/instacart/working_data.csv")
working_data = working_data[working_data['user_id'].isin(test_users)]

working_data.to_csv("/home/ashutosh/Desktop/instacart/test_data.csv")
'''

# step 2.1: converting into usable form
'''
test_data = sc.textFile("/home/ashutosh/Desktop/instacart/test_data.csv")

def parse1(x):
    x = x.split(",")
    y = np.array([0]*len(products))
    y[products.index(int(x[3]))] += 1
    return ((int(x[4]),int(x[5])),y) # user_id, order_number, y

def parse2(x):
    return (x[0][1],(x[0][0],x[1].tolist())) # order_number, user_id, y

def parse3(x):
    return (x[1][0],x[1][1])

test_data = test_data.filter(lambda x: "order_id" not in x and int(x.split(",")[3]) in products).map(parse1).reduceByKey(lambda  x,y: x+y).map(parse2).sortByKey().map(parse3).sortByKey()
test_data = test_data.reduceByKey(lambda x,y:x+[y]).sortByKey()

test_data_format = test_data.collect()
label = ['user_id','input']
test_data_format = pd.DataFrame.from_records(test_data_format,columns=label)

average_orders = pd.read_csv("/home/ashutosh/Desktop/instacart/averageOrders.csv")
test_data_format= test_data_format.merge(average_orders[['user_id','orders','average']],left_on='user_id',right_on='user_id',how='inner')

test_data_format.to_csv("/home/ashutosh/Desktop/instacart/test_data_format.csv")
'''

# step 2.2 : converting it in rnn format
'''
working_data = pd.read_csv("/home/ashutosh/Desktop/instacart/test_data_format.csv")

def formatting1(x):
    x = str(x)
    x = x.replace("[","")
    x = x.replace("]", "")
    x = x.split(",")
    return len(x)/p

working_data['length'] = working_data['input'].apply(lambda x: formatting1(x))

def formatting2(x):
    x = str(x)
    x = x.replace("[","")
    x = x.replace("]", "")
    x = x.split(",")
    a = [[0]*p]*4
    for i in range(min(len(x)/p,4)):
        for j in range(p):
            a[3-i][j] = int(x[-1-j-p*i])
    return a
working_data['input'] = working_data['input'].apply(lambda x: formatting2(x))
del working_data['length']
working_data['input'] = working_data['input'].apply(lambda x: x[0:4])

testUsers = working_data[['user_id','orders','average']]
testUsers.to_csv("/home/ashutosh/Desktop/instacart/testUsers.csv")


x_val = working_data['input']

x_vals_rnn = np.zeros((len(working_data),4,p))

for rows in range(len(working_data)):
    for cols1 in range(4):
        for cols2 in range(p):
            x_vals_rnn[rows][cols1][cols2] = x_val[rows][cols1][cols2]

np.save("/home/ashutosh/Desktop/instacart/x_lstm.npy",x_vals_rnn)

print x_vals_rnn.shape

'''
# creating the file for final test submission
# if not in testUsers, then add the data as in simple deep learning format
'''
order_data = pd.read_csv("/home/ashutosh/Desktop/instacart/orders.csv")
all_test_users = order_data[order_data['eval_set']=="test"]['user_id'].tolist()

lstm_test_users = pd.read_csv("/home/ashutosh/Desktop/instacart/testUsers.csv")['user_id'].tolist()

non_lstm_users = [x for x in all_test_users if x not in lstm_test_users]

non_lstm = pd.read_csv("/home/ashutosh/Desktop/instacart/working_data.csv")
non_lstm = non_lstm[non_lstm['user_id'].isin(non_lstm_users)]

non_lstm.to_csv("/home/ashutosh/Desktop/instacart/non_lstm_data.csv")
'''
# using spark to generate the data file for 63 products for predicting
'''
global products,p

count = pd.read_csv("/home/ashutosh/Desktop/instacart/countOrders.csv")
count = count[count["count"] > 50000]
products = count['product'].tolist()
p = len(products)

non_lstm = sc.textFile("/home/ashutosh/Desktop/instacart/non_lstm_data.csv")

def parse(x):
    x = x.split(",")
    y = np.array([0]*len(products))
    if int(x[3]) in products:
        y[products.index(int(x[3]))] += 1
    return (int(x[4]),y)

non_lstm = non_lstm.filter(lambda x: "order_id" not in x).map(parse).reduceByKey(lambda x,y: x+y).sortByKey()
non_lstm = non_lstm.collect()

label = ['user_id','input']
non_lstm = pd.DataFrame.from_records(non_lstm, columns=label)
average_orders = pd.read_csv("/home/ashutosh/Desktop/instacart/averageOrders.csv")
non_lstm= non_lstm.merge(average_orders[['user_id','orders','average']],left_on='user_id',right_on='user_id',how='inner')

def formatting(x):
    newstr = str(x)
    newstr = newstr.replace("[", "")
    newstr = newstr.replace("]", "")
    newstr = newstr.replace("\n", "")
    newstr = newstr.replace("  ", " ")
    x = newstr.split(" ")
    return [int(x[i]) for i in range(len(products))]

x_non_lstm = non_lstm['input'].apply(lambda x:formatting(x))
x = np.zeros((len(x_non_lstm),p))

for rows in range(len(x_non_lstm)):
    for cols in range(p):
        x[rows][cols] = x_non_lstm[rows][cols]

np.save("/home/ashutosh/Desktop/instacart/x_non_lstm.npy",x)
non_lstm_testUsers = non_lstm[['user_id','orders','average']]
non_lstm_testUsers.to_csv("/home/ashutosh/Desktop/instacart/non_lstm_testUsers.csv")
'''
# generating a better test set for normal deep learning and randomforest data
'''
working_data = pd.read_csv("/home/ashutosh/Desktop/instacart/working_data_format.csv")
working_data = working_data[working_data['orders']>12]

global products,p

count = pd.read_csv("/home/ashutosh/Desktop/instacart/countOrders.csv")
count = count[count["count"] > 25000]
products = count['product'].tolist()
p = len(products)

def formatting1(x):
    x = str(x)
    x = x.replace("[","")
    x = x.replace("]", "")
    x = x.split(",")
    return len(x)

working_data['length'] = working_data['input'].apply(lambda x: formatting1(x))
working_data = working_data[working_data['length']>4*p]

def formatting2(x):
    x = str(x)
    x = x.replace("[","")
    x = x.replace("]", "")
    x = x.split(",")
    a = []
    for i in range(5):
        b = []
        for j in range(len(products)):
            b.append(int(x[-1-j-len(products)*i]))
        a.append(b)
    return a
working_data['input'] = working_data['input'].apply(lambda x: formatting2(x))
del working_data['length']
working_data['output'] = working_data['input'].apply(lambda x: x[-1])
working_data['input'] = working_data['input'].apply(lambda x: x[0:4])

average_orders = pd.read_csv("/home/ashutosh/Desktop/instacart/averageOrders.csv")
working_data= working_data.merge(average_orders[['user_id','average']],left_on='user_id',right_on='user_id',how='inner')

finalUsers = working_data[['user_id','orders','average']]
finalUsers.to_csv("/home/ashutosh/Desktop/instacart/finalUsers.csv")

x_val = working_data['input']
y_val = working_data['output']

x_vals = np.zeros((len(working_data),4,p))
y_vals = np.zeros((len(working_data),p))

for rows in range(len(working_data)):
    for cols in range(p):
        y_vals[rows][cols] = y_val[rows][cols]
    for cols1 in range(4):
        for cols2 in range(p):
            x_vals[rows][cols1][cols2] = x_val[rows][cols1][cols2]

x_vals = np.sum(x_vals,axis=1)
print x_vals.shape
print y_vals.shape
np.save("/home/ashutosh/Desktop/instacart/x_valsN.npy",x_vals)
np.save("/home/ashutosh/Desktop/instacart/y_valsN.npy",y_vals)
'''
