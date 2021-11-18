import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from lib.preprocessor import preprocessor

class PlsaModel():
    def __init__(self, experiment):
        self.experiment = experiment

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
        # self.docs_tf, self.docs_avg = self.docsToVectors(self.docs)
        self.words_docs = (self.wordDocsMatrix(self.docs)).T
        return
        
    def docsToVectors(self, docs):
        docs_weights = []
        docs_avg = 0
        for doc in docs:
            doc_weights, length = self.prep.countTermFrequence(doc)
            docs_avg += length / len(docs)
            docs_weights.append(doc_weights)
        return docs_weights, docs_avg

    def wordDocsMatrix(self, docs):
        docs_weights = []
        docs_avg = 0
        for doc in docs:
            doc_weights, length = self.prep.countTermFrequence(doc)
            docs_avg += length / len(docs)
            docs_weights.append(doc_weights)
        return np.array(docs_weights)

    def initializeParameters(self, docs_length, words_length, topics_length):
        # P(Wi | Tk)
        P_of_wt = np.random.rand(words_length,  topics_length)
        P_of_wt = P_of_wt / np.sum(P_of_wt)
        
        # P(Tk | Dj)
        P_of_td = np.random.rand(topics_length, docs_length)
        P_of_td = P_of_td / np.sum(P_of_td)

        return P_of_wt, P_of_td

    def EM(self, P_of_wt, P_of_td):
        TL, DL = P_of_td.shape
        WL, TL = P_of_wt.shape
        
        # E Step-1
        print('E Step')
        Sum_of_wtd = P_of_wt * P_of_td

        print('M Step')
        for k in tqdm(range(TL)):
            # E Step-2
            P_of_wdt = (np.expand_dims(P_of_wt[:, k], axis=1) * np.expand_dims(P_of_td[k, :], axis=0)) / Sum_of_wtd

            # M Step
            # Update P(Wi | Tk)
            Sum_of_wd_wdt = np.sum(self.words_docs * P_of_wdt, axis=1)
            P_of_wt[:, k] = Sum_of_wd_wdt / (np.sum(Sum_of_wd_wdt, axis=0) + 0.00001)
        
            # Update P(Tk | Dj)
            P_of_td[k, :] = np.sum(self.words_docs * P_of_wdt, axis=0) / (np.sum(self.words_docs, axis=0) + 0.00001)
            
        # Log-likelihood
        print('Log-likelihood')
        Sum_of_wtd = P_of_wt * P_of_td
        
        log_likelihood = np.sum(self.words_docs * np.log(Sum_of_wtd))
        return P_of_wt, P_of_td, log_likelihood

    def train(self, epochs):
        P_of_wt, P_of_td = self.initializeParameters(self.words_docs.shape[1], self.words_docs.shape[0], 30)
        
        print('\nTraining begin.')
        for e in range(epochs):
            print('Epoch {}:'.format(e))
            New_P_of_wt, New_P_of_td, log_likelihood = self.EM(P_of_wt, P_of_td)
            P_of_wt, P_of_td = New_P_of_wt, New_P_of_td
            np.savez('./ckpt/cp_{}.npz'.format(e), P_of_wt=P_of_wt, P_of_td=P_of_td)
            self.experiment.log({
                'Log-likelihood': log_likelihood,
                'step': e
            })
            print('Finished Epoch {}, Log-likelihood: {}'.format(e, log_likelihood), end='\n\n')

    def searchInit(self, alpha=0.5, beta=0.3):
        # load P_of_wt, P_of_td
        data = np.load('./ckpt/1117/cp_33.npz')
        P_of_wt = data['P_of_wt']
        P_of_td = data['P_of_td']

        Sum_of_wtd = np.matmul(P_of_wt, P_of_td)
        norm_of_wd = self.words_docs / np.sum(self.words_docs, axis=0)

        P_of_wbg   = np.sum(self.words_docs, axis=1) / np.sum(self.words_docs)
        P_of_wbg   = np.expand_dims(P_of_wbg, axis=1)
        P_of_wbg   = P_of_wbg * (np.zeros([1, self.words_docs.shape[1]]) + 1)

        tf_plsa = np.logaddexp(np.log(alpha) + np.log(norm_of_wd + 0.000001), \
            np.logaddexp(np.log(beta) + np.log(Sum_of_wtd + 0.000001), np.log(1 - alpha - beta) + np.log(P_of_wbg + 0.000001)))
        self.tf_plsa = tf_plsa
        
    def search(self, query):
        
        query_weights, _ = self.prep.countTermFrequence(query)
        P_of_qw = np.expand_dims(query_weights, axis=0)
        Plsa_result = np.matmul(P_of_qw, self.tf_plsa)

        # first 3000
        result = np.argsort(-Plsa_result)[:3000]
        # print(result)
        return list(result[0, :])