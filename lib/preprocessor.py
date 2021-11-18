import os
import re
import numpy as np
import string as stri

class preprocessor():
    def __init__(self, docs, queries):
        self.inverseTermFrequence, self.termToIndex, self.indexToterm = self.countInverseTermFrequence(docs, queries)
        print('Size of term dics: {}'.format(self.inverseTermFrequence.shape[0]))
    
    def split(self, string):
        terms = list(string.split(" "))
        return terms

    def countInverseTermFrequence(self, docs, queries):
        termCounts = {}
        clean_threshold = 5
        # 建立辭典(只記錄query出現的term)
        # for query in queries:
        #     for term in set(self.split(query)):
        #         termCounts[term] = 0
                    
        for doc in docs:
            for term in set(self.split(doc)):
                if term in termCounts:
                    termCounts[term] += 1
                else:
                    termCounts[term] = 1

        termKeys = list(termCounts.keys())
        for term in termKeys:
            if termCounts[term] < clean_threshold:
                del termCounts[term]
        # 計算idf
        # inverseTermFrequence = np.log(len(docs) / np.array(list(termCounts.values())))
        inverseTermFrequence = (np.log(len(docs) - np.array(list(termCounts.values()))) + 0.5) / (np.array(list(termCounts.values())) + 0.5)
        # inverseTermFrequence = np.power(inverseTermFrequence, 2)
        termToIndex = dict(zip(termCounts.keys(), list(range(len(termCounts)))))
        indexToterm = dict(zip(list(range(len(termCounts))), termCounts.keys()))
        return inverseTermFrequence, termToIndex, indexToterm

    def countTermFrequence(self, string):
        termFrequence = [0] * len(self.termToIndex)
        string = self.split(string)
        for term in string:
            if term not in self.termToIndex: continue
            termFrequence[self.termToIndex[term]] += 1
        tf = np.array(termFrequence)
        # return tf / (np.sum(tf) + 0.0001), len(string)
        return tf, len(string)

    def tf_Idf(self, string):
        tf, length = self.countTermFrequence(string)
        return tf * self.inverseTermFrequence