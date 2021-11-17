import os
import numpy as np

from tqdm import tqdm
from lib.preprocessor import preprocessor

class VectorSpaceModel():
    def __init__(self):
        ...

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
        return docs_dict, queries_dict

    def preprocessing(self, docs, queries):
        self.prep = preprocessor(docs, queries)

    def docsToVectors(self, docs):
        docs_weights = []
        for doc in docs:
            doc_weights = self.prep.tf_Idf(doc)
            docs_weights.append(doc_weights)
        return docs_weights

    def search(self, query, docs_weights):
        query_weights = self.prep.tf_Idf(query)
        result = []
        for index, doc_weights in enumerate(docs_weights):
            rating = np.dot(query_weights, doc_weights) / (np.linalg.norm(doc_weights) * np.linalg.norm(query_weights) + 0.001)
            result.append([index, rating])
        result = sorted(result, key=lambda res: res[1], reverse=True)
        return result