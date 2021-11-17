import os
import numpy as np

from tqdm import tqdm
from lib.preprocessor import preprocessor
import matplotlib.pyplot as plt

class BestMatchModel25():
    def __init__(self):
        self.k1 = 1
        self.b  = 0
        self.history = []
        self.historys= []

    def load(self, docs_paths, queries_paths):
        docs_dict = {}
        print('documents is loading...')
        for filename in tqdm(os.listdir(docs_paths)):
            path = os.path.join(docs_paths, filename)
            with open(path, 'r', encoding='utf-8') as fr:
                docs_dict[filename.split('.txt')[0]] = fr.read()

        queries_dict = {}
        print('queries is loading...')
        for filename in tqdm(os.listdir(queries_paths)):
            path = os.path.join(queries_paths, filename)
            with open(path, 'r', encoding='utf-8') as fr:
                queries_dict[filename.split('.txt')[0]] = fr.read()
        self.docs, self.queries = list(docs_dict.values()), list(queries_dict.values())
        self.preprocessing()
        return docs_dict, queries_dict, self.docs, self.queries

    def preprocessing(self):
        self.prep = preprocessor(self.docs, self.queries)
        self.docs_tf, self.docs_avg = self.docsToVectors(self.docs)
        
    def docsToVectors(self, docs):
        docs_weights = []
        docs_avg = 0
        for doc in docs:
            doc_weights, length = self.prep.countTermFrequence(doc)
            docs_avg += length / len(docs)
            docs_weights.append(doc_weights)
        return docs_weights, docs_avg

    def sim(self, idf, tf, doc_len):
        # 計算Doc與Query之間的Similarity
        score = idf * ((tf * (self.k1 + 1)) / (tf + (self.k1 * (1 - self.b + self.b * (doc_len / self.docs_avg)))))
        self.history.append((doc_len / self.docs_avg))
        return np.sum(score)

    def search(self, query):
        # 計算query的詞頻
        query_tf, _ = self.prep.countTermFrequence(query)
        # query_tf = (query_tf > 0).astype(np.float32)
        query_tf *= 140
        result = []
        # 搜尋所有的Docs
        for index, doc_tf in enumerate(self.docs_tf):
            doc_len = len(self.docs[index])
            termfrequence = query_tf * doc_tf
            rating = self.sim(self.prep.inverseTermFrequence, termfrequence, doc_len)
            result.append([index, rating])
        result = sorted(result, key=lambda res: res[1], reverse=True)
        self.historys.append(np.array(self.history))
        self.history = []
        return result

    def plot(self):
        print(len(self.historys[0]))
        for history in self.historys:
            plt.plot(np.arange(len(history)), history)
        plt.show()
        plt.savefig('history.png')