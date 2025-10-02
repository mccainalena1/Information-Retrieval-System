import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import sklearn.metrics
import  numpy as np
import matplotlib.pyplot as plt
import pprint

### Returns tokenized text in a file (documents or queries)
def processText(fileName):
    # Read text file
    fileObject = open(fileName, "r")
    # Split by document or query
    text = fileObject.read().split(".I ")[1:]
    fileObject.close()
    #Remove newline characters
    # Take text by body only
    # Remove puntuation and extra whitespace
    tokenizedText = [doc[doc.find('.W') + 2:doc.find('Etc')] for doc in text]
    return tokenizedText

# Converts relevancy data into a matrix of documents and queries
def processRelevancyData(fileName):
    # Read file
    relData = np.loadtxt(fileName)
    # Remove extra columns with 0s
    relData = np.delete(relData, 2, 1)
    relData = np.delete(relData, 2, 1)
    relDataMatrix = []
    index = 0
    relCount = relData.shape[0]
    # For each query, make a list of relevant doicuments
    for query in range(1, 226):
        queryDocs = []
        # While the current query is stil found, and it's not the end of the file
        while index < relCount and relData[index][0] == query:
            # Append the relevant document to the list
            queryDocs.append(relData[index][1])
            index += 1
        # Append the list of documents to the list of queries
        relDataMatrix.append(queryDocs)
    return relDataMatrix

# Gets top 10 Euclidian values (by document) for each query
def getTopEuclidean(corpus, queries):
    # Calculate Euclidian distance
    euclideanMatrix = sklearn.metrics.pairwise.pairwise_distances(X=queries, Y=corpus)
    minValues = []
    # For each query
    for query in euclideanMatrix:
        # Get the documents (indicies) of the 10 smallest distaces
        indicies = np.argsort(query)[:10]
        documents = [i + 1 for i in indicies]
        minValues.append(documents)
    return np.array(minValues)

# Gets top 10 Cosine Similarity values (by document) for each query
def getTopCosineSimilarity(corpus, queries):
    cosineSimilarityMatrix = sklearn.metrics.pairwise.cosine_similarity(X=queries, Y=corpus)
    maxValues = []
    # For each query
    for query in cosineSimilarityMatrix:
        # Get the documents (indicies) of the 10 largest similarities
        indicies = np.argsort(query)[-10:]
        documents = [i + 1 for i in indicies]
        maxValues.append(documents)
    return np.array(maxValues)

# Calculate precision for a query based on the actual retrieved documents list, and the relevant documents list
def precision(actual, relevant):
    return np.in1d(actual, relevant).sum() / actual.shape[0]

# Calculate recall for a query based on the actual retrieved documents list, and the relevant documents list
def recall(actual, relevant):
    return np.in1d(actual, relevant).sum() / relevant.shape[0]

# Calculate f for a query based on the precision and recall
def fscore(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)

# Calculate precision, recall, and f for every query
def calcScores(actualMat, relevantMat):
    pData = []
    rData = []
    fData = []
    # For every query (matrix row)
    for query in range(225):
        # Caculate percision, recall, and f
        p = precision(actualMat[query], np.array(relevantMat[query]))
        r = recall(actualMat[query], np.array(relevantMat[query]))
        f = fscore(p, r)
        pData.append(p)
        rData.append(r)
        fData.append(f)
    return pData, rData, fData

# Makes a bar graph wit the specified format
def makeBarGraph(data, title, legend, yLabel):
    plt.bar([i for i in range(1, 226)], data, color='red')
    plt.xticks(ticks=[0, 50, 100, 150, 200], labels=["0", "50", "100", "150", "200"])
    plt.yticks(ticks=[0.0, 0.2, 0.4, 0.6, 0.8], labels=["0.0", "0.2", "0.4", "0.6", "0.8"])
    plt.xlabel("Query Index")
    plt.ylabel(yLabel)
    plt.legend([legend], loc='upper right')
    plt.title(title)
    # UNCOMMENT TO SHOW GRAPHS
    # plt.show()

# Makes Dictionary of statistics in the specified format
def calcStats(euclidianP, euclidianR, euclidianF, cosineSimilarityP, cosineSimilarityR, cosineSimilarityF):
    fMetricDict = {"cos": ((sum(cosineSimilarityF) / 225), max(cosineSimilarityF)), "euc": ((sum(euclidianF) / 225), max(euclidianF))}
    pMetricDict = {"cos": ((sum(cosineSimilarityP) / 225), max(cosineSimilarityP)), "euc": ((sum(euclidianP) / 225), max(euclidianP))}
    rMetricDict = {"cos": ((sum(cosineSimilarityR) / 225), max(cosineSimilarityR)), "euc": ((sum(euclidianR) / 225), max(euclidianR))}
    return {"f" : fMetricDict, "p" : pMetricDict, "r" : rMetricDict} 

# Does all calculations for Binary model
def runBinary(tokenizedCorpus, tokenizedQueries, relevantMatrix):
    # Create and use vectorizer
    binaryVectorizer = CountVectorizer(lowercase=True, stop_words=list(text.ENGLISH_STOP_WORDS), binary=True)
    corpusBinary = binaryVectorizer.fit_transform(tokenizedCorpus)
    queryBinary = binaryVectorizer.transform(tokenizedQueries)
    # Get top 10 documents for each query based on Euclidean distance and Cosine Similarity
    topEuclidian = getTopEuclidean(corpusBinary, queryBinary)
    topCosineSimilarity = getTopCosineSimilarity(corpusBinary, queryBinary)
    # Calculate precision, recall, and F for each query based on Euclidean distance and Cosine Similarity
    euclidianP, euclidianR, euclidianF = calcScores(topEuclidian, relevantMatrix)
    cosineSimilarityP, cosineSimilarityR, cosineSimilarityF = calcScores(topCosineSimilarity, relevantMatrix)
    # Make bar graphs of precision, recall, and F for each query based on Euclidean distance and Cosine Similarity
    makeBarGraph(euclidianP, "Precision of each query (10 most relavant documents) - Binary\n(using euclidian distance)", "Binary", "Precision")
    makeBarGraph(cosineSimilarityP, "Precision of each query (10 most relavant documents) - Binary\n(using cosine similarity)", "Binary", "Precision")
    makeBarGraph(euclidianR, "Recall of each query (10 most relavant documents) - Binary\n(using euclidian distance)", "Binary", "Recall")
    makeBarGraph(cosineSimilarityR, "Recall of each query (10 most relavant documents) - Binary\n(using cosine similarity)", "Binary", "Recall")
    makeBarGraph(euclidianF, "F of each query (10 most relavant documents) - Binary\n(using euclidian distance)", "Binary", "F")
    makeBarGraph(cosineSimilarityF, "F of each query (10 most relavant documents) - Binary\n(using cosine similarity)", "Binary", "F")
    # Put Binary model statistics of precision, recall, and F for based on Euclidean distance and Cosine Similarity into dictionary
    return calcStats(euclidianP, euclidianR, euclidianF, cosineSimilarityP, cosineSimilarityR, cosineSimilarityF)

# Does all calculations for TFIDF model
def runTFIDF(tokenizedCorpus, tokenizedQueries, relevantMatrix):
    # Create and use vectorizer
    tfidfVectorizer = TfidfVectorizer(lowercase=True, stop_words=list(text.ENGLISH_STOP_WORDS))
    corpusTdif = tfidfVectorizer.fit_transform(tokenizedCorpus)
    queryTdif = tfidfVectorizer.transform(tokenizedQueries)
    # Get top 10 documents for each query based on Euclidean distance and Cosine Similarity
    topEuclidian = getTopEuclidean(corpusTdif, queryTdif)
    topCosineSimilarity = getTopCosineSimilarity(corpusTdif, queryTdif)
    # Calculate precision, recall, and F for each query based on Euclidean distance and Cosine Similarity
    euclidianP, euclidianR, euclidianF = calcScores(topEuclidian, relevantMatrix)
    cosineSimilarityP, cosineSimilarityR, cosineSimilarityF = calcScores(topCosineSimilarity, relevantMatrix)
    # Make bar graphs of precision, recall, and F for each query based on Euclidean distance and Cosine Similarity
    makeBarGraph(euclidianP, "Precision of each query (10 most relavant documents) - TF-IDF\n(using euclidian distance)", "TF-IDF", "Precision")
    makeBarGraph(cosineSimilarityP, "Precision of each query (10 most relavant documents) - TF-IDF\n(using cosine similarity)", "TF-IDF", "Precision")
    makeBarGraph(euclidianR, "Recall of each query (10 most relavant documents) - TF-IDF\n(using euclidian distance)", "TF-IDF", "Recall")
    makeBarGraph(cosineSimilarityR, "Recall of each query (10 most relavant documents) - TF-IDF\n(using cosine similarity)", "TF-IDF", "Recall")
    makeBarGraph(euclidianF, "F of each query (10 most relavant documents) - TF-IDF\n(using euclidian distance)", "TF-IDF", "F")
    makeBarGraph(cosineSimilarityF, "F of each query (10 most relavant documents) - TF-IDF\n(using cosine similarity)", "TF-IDF", "F")
    # Put TFIDF model statistics of precision, recall, and F for based on Euclidean distance and Cosine Similarity into dictionary
    return calcStats(euclidianP, euclidianR, euclidianF, cosineSimilarityP, cosineSimilarityR, cosineSimilarityF)

################### MAIN CODE ############################
# Read in files
tokenizedCorpus = processText("./CranfieldDataset/cran.all")
tokenizedQueries = processText("./CranfieldDataset/query.text")
relevantMatrix = processRelevancyData("./CranfieldDataset/qrels.text")
# Create vectorizers, calculate metris, create graphs, and calculate stats
binaryDict = runBinary(tokenizedCorpus, tokenizedQueries, relevantMatrix)
tfidfDict  = runTFIDF(tokenizedCorpus, tokenizedQueries, relevantMatrix)
# Format stats for output
statsDict = {"Binary": binaryDict, "TFIDF": tfidfDict}
pp = pprint.PrettyPrinter()
pp.pprint(statsDict)