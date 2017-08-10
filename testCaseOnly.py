import os
import sys
import pandas as pd
import numpy as np

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

productCO = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/CO.csv")
products = np.array(productCO['product_id'].tolist()[0:10])

'''
testUsers = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/submit.csv")['user_id'].tolist()
temp = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/WDnew.csv")
temp = temp[temp['user_id'].isin(testUsers)]
temp.to_csv("/home/ashutosh/Desktop/instacart/testcase/temp1.csv")

temp = sc.textFile("/home/ashutosh/Desktop/instacart/testcase/temp1.csv")
def parse1(x):
    x = str(x)
    x = x.split(",")
    return ((int(x[6]),int(x[5])),1)
def parse2(x):
    return (x[0][0],x[0][1],x[1])
temp = temp.filter(lambda x: "order_id" not in x).map(parse1).reduceByKey(lambda x,y: x+y).map(parse2)
temp = temp.collect()
label = ['user_id','product_id','count']
temp = pd.DataFrame.from_records(temp,columns=label)

temp.to_csv("/home/ashutosh/Desktop/instacart/testcase/temp2.csv")

temp = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/temp2.csv")
temp = temp.sort_values(by=['user_id', 'count'], ascending=[1, 0])

users = list(set(temp['user_id'].tolist()))
for i in range(len(users)):
    tempo = temp[temp['user_id']==users[i]]
    tempoP = tempo['product_id'].tolist()[0:2]
    countP = tempo['count'].tolist()[0:2]
    for j in range(1):
        if tempoP[j] not in products and countP[j] > 3:
            products = np.append(products,tempoP[j])

np.save("/home/ashutosh/Desktop/instacart/testcase/products.npy",products)
print products.shape
'''
products = np.load("/home/ashutosh/Desktop/instacart/testcase/products.npy")
products = products.tolist()
'''
testUsers = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/submit.csv")['user_id'].tolist()
train_test = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/WDnew.csv")
train_test = train_test[train_test['user_id'].isin(testUsers)]
train_test = train_test.sort_values(by=['user_id', 'order_number'], ascending=[1, 0])

y_orders = train_test.groupby('user_id').first()['order_id'].tolist()
y_data = train_test[train_test['order_id'].isin(y_orders)]
x_data = train_test[~train_test['order_id'].isin(y_orders)]
y_data.to_csv("/home/ashutosh/Desktop/instacart/testcase/y_train_test.csv")
x_data.to_csv("/home/ashutosh/Desktop/instacart/testcase/x_train_test.csv")

y_data = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/y_train_test.csv")
x_data = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/x_train_test.csv")

cols = [0,1,2,3,7]
y_data = y_data.drop(y_data.columns[cols],axis = 1)
x_data = x_data.drop(x_data.columns[cols],axis = 1)
y_data.to_csv("/home/ashutosh/Desktop/instacart/testcase/y_train_test.csv")
x_data.to_csv("/home/ashutosh/Desktop/instacart/testcase/x_train_test.csv")

'''
# creating the data manually without spark

y_data = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/y_train_test.csv")
x_data = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/x_train_test.csv")

y_data = y_data[y_data['product_id'].isin(products)]
x_data = x_data[x_data['product_id'].isin(products)]

testUsers = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/submit.csv")['user_id'].tolist()

Y = np.zeros((len(testUsers),len(products)))
X = np.zeros((len(testUsers),len(products)))

for i in range(len(x_data)):
    u = testUsers.index(int(x_data.iloc[i,3]))
    p = products.index(int(x_data.iloc[i,2]))
    X[u][p] += 1
    print i, len(x_data)

np.save("/home/ashutosh/Desktop/instacart/testcase/X.npy",X)

'''
working_data = sc.textFile("/home/ashutosh/Desktop/instacart/testcase/y_train_test.csv")

def parse1(x):
    x = x.split(",")
    y = np.array([0]*len(products))
    y[products.index(int(x[5]))] += 1
    return (int(x[6]),y) # user_id, order_number, y

working_data = working_data.filter(lambda x: "order_id" not in x and int(x.split(",")[5]) in products).map(parse1).reduceByKey(lambda  x,y: x+y).sortByKey()
working_data_format = working_data.collect()
label = ['user_id','response']
working_data_format = pd.DataFrame.from_records(working_data_format,columns=label)
working_data_format.to_csv("/home/ashutosh/Desktop/instacart/testcase/response.csv")
print len(working_data), working_data.head()


# creating the predictor variables
print uuu
testUsers = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/submit.csv")['user_id'].tolist()
train_test = pd.read_csv("/home/ashutosh/Desktop/instacart/testcase/WDnew.csv")
train_test = train_test[train_test['user_id'].isin(testUsers)]
train_test = train_test.sort_values(by=['user_id', 'order_number'], ascending=[1, 0])

y_orders = train_test.groupby('user_id').first()['order_id'].tolist()
x_data = train_test[~train_test['order_id'].isin(y_orders)]

x_data.to_csv("/home/ashutosh/Desktop/instacart/testcase/x_train_test.csv")


print uuu

working_data_format = sc.textFile("/home/ashutosh/Desktop/instacart/testcase/working_data_format.csv")
def parse1(x):
    x = x.split(",")
    y = np.array([0]*len(products))
    y[products.index(int(x[5]))] += 1
    return ((int(x[6]),int(x[7])),y) # user_id, order_number, y

def parse2(x):
    return (x[0][1],(x[0][0],x[1].tolist())) # order_number, user_id, y

def parse3(x):
    return (x[1][0],x[1][1],x[0])

working_data = working_data.filter(lambda x: "order_id" not in x and int(x.split(",")[4]) in products).map(parse1).reduceByKey(lambda  x,y: x+y).map(parse2).sortByKey().map(parse3).reduceByKey(lambda x,y:x+y)

working_data_format = working_data.collect()
label = ['user_id','input','order_number']
working_data_format = pd.DataFrame.from_records(working_data_format,columns=label)
working_data_format.to_csv("/home/ashutosh/Desktop/instacart/testcase/working_data_format.csv")


average_orders = pd.read_csv("/home/ashutosh/Desktop/instacart/averageOrders.csv")
working_data_format = working_data_format.merge(average_orders[['user_id','orders','average']],left_on='user_id',right_on='user_id',how='inner')

working_data_format.to_csv("/home/ashutosh/Desktop/instacart/testcase/working_data_format.csv")
'''