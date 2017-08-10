import os
import sys
import re
from math import sqrt

# Path for spark source folder
os.environ['SPARK_HOME']="/home/ashutosh/Dropbox/spark-2.1.1-bin-hadoop2.7"

# Append pyspark  to Python Path
sys.path.append("/home/ashutosh/Dropbox/spark-2.1.1-bin-hadoop2.7/python")
sys.path.append("/home/ashutosh/Dropbox/spark-2.1.1-bin-hadoop2.7/python/lib/py4j-0.10.4-src.zip")


try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import SparkSession
    from pyspark.sql import Row
    from pyspark.sql import functions
    from pyspark.mllib.recommendation import ALS, Rating

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

conf = SparkConf().setMaster("local[*]").setAppName("My App")
sc = SparkContext(conf=conf)


# movie rating program: rating counter
import  collections

lines = sc.textFile("/home/ashutosh/Desktop/datasetSpark/ratings.csv")
ratings = lines.map(lambda x: x.split(",")[2])
results = ratings.countByValue()

sortedResults = collections.OrderedDict(sorted(results.items()))
for key,value in sortedResults.iteritems():
    print "%s %i"% (key,value)

# -----------------------------------------------
'''
# key value pair example
# lambda x: (x,1), groupBYKey, sortByKey,reduceByKey
# can also combine two key value pairs for inner join, outer join etc
def ParseValue(lines):
    field = lines.split(",")
    age = int(field[2])
    friends = int(field[3])
    return (age, friends)

lines = sc.textFile("/home/ashutosh/Desktop/datasetSpark/fakefriends.csv")
rdd = lines.map(ParseValue)
# converting (33,34) to (33,(34,1)) to help keep a count: key will remain as it is
# the following operation is done on values of the key value pair (thus x,y are the value)
totalValue = rdd.mapValues(lambda  x: (x,1)).reduceByKey(lambda x,y:(x[0]+y[0],x[1]+y[1]))
# the following converts x, (y1,y2) to x,y
average = totalValue.mapValues(lambda x:x[0]/x[1])
results = average.collect()
for result in results:
    print result
'''
# -----------------------------------------------
'''
# learning the filter function
def parseLine(line):
    fields = line.split(',')
    stationID = fields[0]
    entryType = fields[2]
    temperature = float(fields[3]) * 0.1 * (9.0 / 5.0) + 32.0
    return (stationID, entryType, temperature)

lines = sc.textFile("/home/ashutosh/Desktop/datasetSpark/1800.csv")
parsedLines = lines.map(parseLine)
minTemps = parsedLines.filter(lambda x: "TMIN" in x[1])
stationTemps = minTemps.map(lambda x: (x[0], x[2]))
minTemps = stationTemps.reduceByKey(lambda x, y: min(x,y))
results = minTemps.collect()

for result in results:
    print(result[0] + "\t{:.2f}F".format(result[1]))
'''
# -----------------------------------------------------
'''
# learning flatMap() function
# flatMap converts (a b, c d) to (a,b,c,d) like flatten function in python

# -- naive version
# lines = sc.textFile("/home/ashutosh/Desktop/datasetSpark/Book.txt")
# words = lines.flatMap(lambda x: x.split())
# wordCount = words.countByValue()

# improved version
def splitting(lines):
    return re.compile(r'\W+', re.UNICODE).split(lines.lower())

lines = sc.textFile("/home/ashutosh/Desktop/datasetSpark/Book.txt")
words = lines.flatMap(splitting)
wordCount = words.countByValue()

# for word, count in wordCount.items():
#     cleanWord = word.encode('ascii', 'ignore')
#     if (cleanWord):
#         print(cleanWord.decode() + " " + str(count))

# sorting out the results

wordCounts = words.map(lambda x: (x,1)).reduceByKey(lambda x,y:x+y)
valueSorted = wordCounts.map(lambda x:(x[1],x[0])).sortByKey()

results = valueSorted.collect()

for result in results:
    count = str(result[0])
    word = result[1].encode('ascii', 'ignore')
    if (word):
        print(word.decode() + ":\t\t" + count)

'''
# ---------------------------------------------------------
# flipping the key value pair
# rdd = rdd.map(lambda (x,y): (y,x))

# Assignment 1
'''
def parseLine(line):
    fields = line.split(',')
    customerID = int(fields[0])
    amount = float(fields[2])
    return (customerID,amount)

lines = sc.textFile("/home/ashutosh/Desktop/datasetSpark/customer-orders.csv")
parsedLines = lines.map(parseLine)

amount = parsedLines.reduceByKey(lambda x,y: x+y)
results = amount.collect()

for result in results:
    print result[0], result[1]

sortByAmount = amount.map(lambda x:(x[1],x[0])).sortByKey()
sortResults = sortByAmount.collect()


for result in sortResults:
    print result[0], result[1]

'''

# ------------------------------------------------------------------
# manipulation on movie data
# learning broadcast variables
# .broadcast to ship the value to the executor nodes
# .value to get back the object
'''
def loadMovieNames():
    # creating a dictionary for movie ID and movie name
    movieNames = {}
    with open("/home/ashutosh/Desktop/datasetSpark/ml-100k/u.item") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

nameDict = sc.broadcast(loadMovieNames())

lines = sc.textFile("/home/ashutosh/Desktop/datasetSpark/ml-100k/u.data")
movies = lines.map(lambda x: (int(x.split()[1]), 1))
movieCounts = movies.reduceByKey(lambda x, y: x + y)

flipped = movieCounts.map( lambda x : (x[1], x[0]))
sortedMovies = flipped.sortByKey()

sortedMoviesWithNames = sortedMovies.map(lambda countMovie : (nameDict.value[countMovie[1]], countMovie[0]))

results = sortedMoviesWithNames.collect()

for result in results:
    print result

'''
# ----------------------------------
'''
# super hero network analysis

def countCoOccurences(line):
    # counts the number of cooccurences
    elements = line.split()
    return (int(elements[0]), len(elements) - 1)

def parseNames(line):
    # makes a key value pair for superhero id and name
    fields = line.split('\"')
    return (int(fields[0]), fields[1].encode("utf8"))

names = sc.textFile("/home/ashutosh/Desktop/datasetSpark/Marvel-Names.txt")
namesRdd = names.map(parseNames)

lines = sc.textFile("/home/ashutosh/Desktop/datasetSpark/Marvel-Graph.txt")

pairings = lines.map(countCoOccurences)
totalFriendsByCharacter = pairings.reduceByKey(lambda x, y : x + y)
flipped = totalFriendsByCharacter.map(lambda xy : (xy[1], xy[0]))

# finds the maximum with respect to the key
mostPopular = flipped.max()

mostPopularName = namesRdd.lookup(mostPopular[1])[0]

print(str(mostPopularName) + " is the most popular superhero, with " + \
    str(mostPopular[0]) + " co-appearances.")

'''

# --------------------------------------------------------
# learning accumulators
# learning breadth first search
# very important: lecture 23
# convert the problem into an framework that can be solved by spark
'''
# The characters we wish to find the degree of separation between:
startCharacterID = 5306 #SpiderMan
targetCharacterID = 14  #ADAM 3,031

hitCounter = sc.accumulator(0)

def convertToBFS(line):
    fields = line.split()
    heroID = int(fields[0])
    connections = []
    for connection in fields[1:]:
        connections.append(int(connection))

    color = 'WHITE'
    distance = 9999

    if (heroID == startCharacterID):
        color = 'GRAY'
        distance = 0

    return (heroID, (connections, distance, color))

def createStartingRdd():
    inputFile = sc.textFile("/home/ashutosh/Desktop/datasetSpark/Marvel-Graph.txt")
    return inputFile.map(convertToBFS)

def bfsMap(node):
    characterID = node[0]
    data = node[1]
    connections = data[0]
    distance = data[1]
    color = data[2]

    results = []

    if (color == 'GRAY'):
        for connection in connections:
            newCharacterID = connection
            newDistance = distance + 1
            newColor = 'GRAY'
            if (targetCharacterID == connection):
                hitCounter.add(1)

            newEntry = (newCharacterID, ([], newDistance, newColor))
            results.append(newEntry)

        color = 'BLACK'  # because the node has been processed

    results.append((characterID, (connections, distance, color)))
    return results

def bfsReduce(data1, data2):
    edges1 = data1[0]
    edges2 = data2[0]
    distance1 = data1[1]
    distance2 = data2[1]
    color1 = data1[2]
    color2 = data2[2]

    distance = 9999
    color = color1
    edges = []

    # See if one is the original node with its connections.
    # If so preserve them.
    if (len(edges1) > 0):
        edges.extend(edges1)
    if (len(edges2) > 0):
        edges.extend(edges2)

    # Preserve minimum distance
    if (distance1 < distance):
        distance = distance1

    if (distance2 < distance):
        distance = distance2

    # Preserve darkest color
    if (color1 == 'WHITE' and (color2 == 'GRAY' or color2 == 'BLACK')):
        color = color2

    if (color1 == 'GRAY' and color2 == 'BLACK'):
        color = color2

    if (color2 == 'WHITE' and (color1 == 'GRAY' or color1 == 'BLACK')):
        color = color1

    if (color2 == 'GRAY' and color1 == 'BLACK'):
        color = color1

    return (edges, distance, color)

iterationRdd = createStartingRdd()

for iteration in range(0, 10):
    print("Running BFS iteration# " + str(iteration+1))

    # Create new vertices as needed to darken or reduce distances in the
    # reduce stage. If we encounter the node we're looking for as a GRAY
    # node, increment our accumulator to signal that we're done.
    mapped = iterationRdd.flatMap(bfsMap)

    # Note that mapped.count() action here forces the RDD to be evaluated, and
    # that's the only reason our accumulator is actually updated.
    print("Processing " + str(mapped.count()) + " values.")

    if (hitCounter.value > 0):
        print("Hit the target character! From " + str(hitCounter.value) \
            + " different direction(s).")
        break

    # Reducer combines data for each character ID, preserving the darkest
    # color and shortest path.
    iterationRdd = mapped.reduceByKey(bfsReduce)
'''

# ------------------------------------------------------------------------------
# learning more complex problem in spark
# idea is to cache a RDD if it is to be used multiple times
# lecture 25 very important

'''

def loadMovieNames():
    movieNames = {}
    with open("/home/ashutosh/Desktop/datasetSpark/ml-100k/u.item") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

#Python 3 doesn't let you pass around unpacked tuples,
#so we explicitly extract the ratings now.
def makePairs( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))

def filterDuplicates( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2

def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)


print("\nLoading movie names...")
nameDict = loadMovieNames()

data = sc.textFile("/home/ashutosh/Desktop/datasetSpark/ml-100k/u.data")

# Map ratings to key / value pairs: user ID => movie ID, rating
ratings = data.map(lambda l: l.split()).map(lambda l: (int(l[0]), (int(l[1]), float(l[2]))))

# Emit every movie rated together by the same user.
# Self-join to find every combination.
joinedRatings = ratings.join(ratings)

# At this point our RDD consists of userID => ((movieID, rating), (movieID, rating))

# Filter out duplicate pairs
uniqueJoinedRatings = joinedRatings.filter(filterDuplicates)

# Now key by (movie1, movie2) pairs.
moviePairs = uniqueJoinedRatings.map(makePairs)

# We now have (movie1, movie2) => (rating1, rating2)
# Now collect all ratings for each movie pair and compute similarity
moviePairRatings = moviePairs.groupByKey()

# We now have (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
# Can now compute similarities.
# mapValues maintain the keys of the object
moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache()

# Save the results if desired
#moviePairSimilarities.sortByKey()
#moviePairSimilarities.saveAsTextFile("movie-sims")

# Extract similarities for the movie we care about that are "good".

if (len(sys.argv) > 0):

    scoreThreshold = 0.97
    coOccurenceThreshold = 50

    movieID = 53#int(sys.argv[1])

    # Filter for movies with this sim that are "good" as defined by
    # our quality thresholds above
    filteredResults = moviePairSimilarities.filter(lambda pairSim: \
        (pairSim[0][0] == movieID or pairSim[0][1] == movieID) \
        and pairSim[1][0] > scoreThreshold and pairSim[1][1] > coOccurenceThreshold)

    # Sort by quality score.
    results = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(10)

    print("Top 10 similar movies for " + nameDict[movieID])
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = pair[0]
        if (similarMovieID == movieID):
            similarMovieID = pair[1]
        print(nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))

#'''
# ----------------------------------------------------------
# localhost:4040/jobs tells us about the status of the jobs
# spark SQL
'''

import collections

# Create a SparkSession (Note, the config section is only for Windows!)
spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("SparkSQL").getOrCreate()

def mapper(line):
    # providing a more structured wat: imparting structure
    fields = line.split(',')
    return Row(ID=int(fields[0]), name=str(fields[1].encode("utf-8")), age=int(fields[2]), numFriends=int(fields[3]))

lines = spark.sparkContext.textFile("fakefriends.csv")
people = lines.map(mapper)

# Infer the schema, and register the DataFrame as a table.
schemaPeople = spark.createDataFrame(people).cache()
schemaPeople.createOrReplaceTempView("people")

# SQL can be run over DataFrames that have been registered as a table.
teenagers = spark.sql("SELECT * FROM people WHERE age >= 13 AND age <= 19")

# The results of SQL queries are RDDs and support all the normal RDD operations.
for teen in teenagers.collect():
  print(teen)

# We can also use functions instead of SQL queries:
schemaPeople.groupBy("age").count().orderBy("age").show()

spark.stop()

'''
# ------------------------------------------------------------
# learning using the dataframes instead of RDDs

'''

def loadMovieNames():
    movieNames = {}
    with open("/home/ashutosh/Desktop/datasetSpark/ml-100k/u.item") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

# Load up our movie ID -> name dictionary
nameDict = loadMovieNames()

# Create a SparkSession (the config bit is only for Windows!)
spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("PopularMovies").getOrCreate()

# Get the raw data
lines = spark.sparkContext.textFile("/home/ashutosh/Desktop/datasetSpark/ml-100k/u.data")
# Convert it to a RDD of Row objects
movies = lines.map(lambda x: Row(movieID =int(x.split()[1])))
# Convert that to a DataFrame
movieDataset = spark.createDataFrame(movies)

# Some SQL-style magic to sort all movies by popularity in one line!
topMovieIDs = movieDataset.groupBy("movieID").count().orderBy("count", ascending=False).cache()

# Show the results at this point:

#|movieID|count|
#+-------+-----+
#|     50|  584|
#|    258|  509|
#|    100|  508|

topMovieIDs.show()

# Grab the top 10
top10 = topMovieIDs.take(10)

# Print the results
print("\n")
for result in top10:
    # Each row has movieID, count as above.
    print("%s: %d" % (nameDict[result[0]], result[1]))

# Stop the session, this connects to the database
spark.stop()

'''

# --------- till now, we learnt on spark core
# now we learn the different libraries of the spark

'''
# recommending movie on MLLIB based on alternating least squares
def loadMovieNames():
    movieNames = {}
    with open("ml-100k/u.ITEM", encoding='ascii', errors="ignore") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

conf = SparkConf().setMaster("local[*]").setAppName("MovieRecommendationsALS")
sc = SparkContext(conf = conf)
sc.setCheckpointDir('checkpoint')

print("\nLoading movie names...")
nameDict = loadMovieNames()

data = sc.textFile("file:///SparkCourse/ml-100k/u.data")

ratings = data.map(lambda l: l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()

# Build the recommendation model using Alternating Least Squares
print("\nTraining recommendation model...")
rank = 10
# Lowered numIterations to ensure it works on lower-end systems
numIterations = 6
model = ALS.train(ratings, rank, numIterations)

userID = int(sys.argv[1])

print("\nRatings for user ID " + str(userID) + ":")
userRatings = ratings.filter(lambda l: l[0] == userID)
for rating in userRatings.collect():
    print (nameDict[int(rating[1])] + ": " + str(rating[2]))

print("\nTop 10 recommendations:")
recommendations = model.recommendProducts(userID, 10)
for recommendation in recommendations:
    print (nameDict[int(recommendation[1])] + \
        " score " + str(recommendation[2]))

'''