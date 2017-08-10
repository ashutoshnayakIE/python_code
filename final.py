import os
import sys
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
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

# developing the correlation effect of the products
'''
total = [0]*12012

TP = np.load("/home/ashutosh/Desktop/instacart/final/corr.npy")
for i in range(12012):
    TP[i][i] /= 2
    total[i] = TP[i][i]

allTotal = np.sum(TP)
corr = np.zeros((12012,12012))

for i in range(12012):
    for j in range(12012):
        tp = TP[i][j]
        tn = allTotal+TP[j][i]-total[i]-total[j]
        fp = total[j]-TP[i][j]
        fn = total[i]-TP[j][i]
        c = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
        corr[j][i] = (tp*tn - fp*fn)/math.sqrt(c)

np.save("/home/ashutosh/Desktop/instacart/final/correlation.npy",corr)
'''
# ===============================================================


corr = np.load("/home/ashutosh/Desktop/instacart/final/correlation.npy")
for i in range(12012):
    sump = sum(corr[:,i])
    corr[:,i] /= sump

np.save("/home/ashutosh/Desktop/instacart/final/correlation1.npy",corr)

print sum(corr[0]),sum(corr[:,0]),uuu
submit= pd.read_csv("/home/ashutosh/Desktop/instacart/final/submit.csv")
products= pd.read_csv("/home/ashutosh/Desktop/instacart/final/timeStamp.csv")['product_id'].tolist()
users = submit['user_id'].tolist()
count = submit['average'].tolist()

countOrders = pd.read_csv("/home/ashutosh/Desktop/instacart/final/countOrders.csv")
countOrders = countOrders[countOrders['product'].isin(products)]
tO = sum(countOrders['count'].tolist())
pr = [0]*12012
for i in range(len(countOrders)):
    p = products.index(int(countOrders.iloc[i,0]))
    pr[p] = int(countOrders.iloc[i,1])/float(tO)

submitTry = pd.read_csv("/home/ashutosh/Desktop/instacart/final/best.csv")

X = np.load("/home/ashutosh/Desktop/instacart/final/X1.npy")
P = np.load("/home/ashutosh/Desktop/instacart/final/P_new.npy")
Ttest = np.load("/home/ashutosh/Desktop/instacart/final/Ttest.npy")[0:15000]
corr = np.load("/home/ashutosh/Desktop/instacart/final/correlation.npy")

'''
I = np.ones((12012, 12012))
np.fill_diagonal(I, 1)
corr = np.multiply(corr, I)
'''
fprob = np.dot(X, corr)
tprob = np.dot(Ttest,P)
fprob = np.multiply(fprob,tprob)

fprob += 0.001*pr

for i in range(15000):
    prob = np.argsort(fprob[i])[::-1]
    s = ""
    for t in range(int(count[i])-1):
        s += str(products[prob[t]])
        if t < int(count[i]) - 1-1:
            s += " "
    submitTry.iloc[i, 1] = s
    print i

submitTry.to_csv("/home/ashutosh/Desktop/instacart/final/submitTry.csv")

print uuu

for y in range(5,6):

    submit= pd.read_csv("/home/ashutosh/Desktop/instacart/final/submit.csv")
    products= pd.read_csv("/home/ashutosh/Desktop/instacart/final/timeStamp.csv")['product_id'].tolist()
    users = submit['user_id'].tolist()[15000*(y-1):15000*y]
    count = submit['average'].tolist()[15000*(y-1):15000*y]

    data = pd.read_csv("/home/ashutosh/Desktop/instacart/final/dataTopP.csv")
    data = data.sort_values(by=['user_id', 'order_number'], ascending=[1, 0])
    data = data[data['user_id'].isin(users)]
    data = data.groupby('order_id').first()
    data.to_csv("/home/ashutosh/Desktop/instacart/final/Y"+str(y)+".csv")

    Y = np.zeros((15000,12012))
    for i in range(len(data)):
        p = products.index(int(data.iloc[i,4]))
        u = users.index(int(data.iloc[i,5]))
        Y[u][p] += 1

    np.save("/home/ashutosh/Desktop/instacart/final/Y"+str(y)+".npy",Y)

print uuu
submit= pd.read_csv("/home/ashutosh/Desktop/instacart/final/submit.csv")
products= pd.read_csv("/home/ashutosh/Desktop/instacart/final/timeStamp.csv")['product_id'].tolist()
users = submit['user_id'].tolist()
count = submit['average'].tolist()

data = pd.read_csv("/home/ashutosh/Desktop/instacart/final/dataTopP.csv")
data = data[data['user_id']==36855]
data = data[data['order_number'] < 4]
pr = data['product_id'].tolist()
X = [0]*12012
for i in range(len(pr)):
    X[products.index(pr[i])] += 1

submitTry = pd.read_csv("/home/ashutosh/Desktop/instacart/final/best.csv")

X = np.load("/home/ashutosh/Desktop/instacart/final/X1.npy")
#P = np.load("/home/ashutosh/Desktop/instacart/final/P_new.npy")
#Ttest = np.load("/home/ashutosh/Desktop/instacart/final/Ttest.npy")[0:15000]
corr = np.load("/home/ashutosh/Desktop/instacart/final/corr.npy")
for i in range(12012):
    corr[i][i] /= 2
    sump = sum(corr[:,i])
    corr[:,i] /= sump
'''
I = np.ones((12012, 12012))
np.fill_diagonal(I, 1)
corr = np.multiply(corr, I)
'''
fprob = np.dot(X, corr)
# tprob = np.dot(Ttest,P)
# fprob = np.multiply(pprob,tprob)

for i in range(15000):
    prob = np.argsort(fprob[i])[::-1]
    s = ""
    for t in range(int(count[i])):
        s += str(products[prob[t]])
        if t < int(count[i]) - 1:
            s += " "
    submitTry.iloc[i, 1] = s

submitTry.to_csv("/home/ashutosh/Desktop/instacart/final/submitTry.csv")