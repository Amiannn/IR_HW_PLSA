import os
import wandb
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from lib.vsm  import VectorSpaceModel
from lib.bm25 import BestMatchModel25
from lib.plsa import PlsaModel

def vsm(docs_path, queries_path):

    vsm = VectorSpaceModel()
    
    docs_dict, queries_dict = vsm.load(docs_path, queries_path)
    docs, queries = list(docs_dict.values()), list(queries_dict.values())
    
    with open('queries.txt', 'w', encoding='utf-8') as fr:
        for q in sorted(queries):
            fr.write('{}\n'.format(q))

    vsm.preprocessing(docs, queries)

    with open('term_dict.txt', 'w', encoding='utf-8') as fr:
        terms = sorted(list(vsm.prep.indexToterm.values()))
        print('making term dict...')
        for term in tqdm(terms):
            fr.write('{},\t{}\n'.format(term, vsm.prep.inverseTermFrequence[vsm.prep.termToIndex[term]]))

    docs_keys   = list(docs_dict.keys())
    docs_weights= vsm.docsToVectors(docs)

    with open('vsm_result.txt', 'w', encoding='utf-8') as fr:
        fr.write('Query,RetrievedDocuments\n')
        print('searching thought queries...')
        for index in tqdm(queries_dict):
            result = vsm.search(queries_dict[index], docs_weights)
            fr.write('{}, '.format(index))
            for res in result:
                fr.write('{} '.format(docs_keys[res[0]]))
            fr.write('\n')

def bm25(docs_path, queries_path):
    BM25 = BestMatchModel25()
    docs_dict, queries_dict, docs, queries = BM25.load(docs_path, queries_path)
    docs_keys   = list(docs_dict.keys())

    with open('result.txt', 'w', encoding='utf-8') as fr:
        fr.write('Query,RetrievedDocuments\n')
        print('searching thought queries...')
        for index in tqdm(queries_dict):
            result = BM25.search(queries_dict[index])
            fr.write('{}, '.format(index))
            for res in result:
                fr.write('{} '.format(docs_keys[res[0]]))
            fr.write('\n')

def plsa(docs_path, queries_path, epochs, experiment):
    PLSA = PlsaModel(experiment)
    docs_dict, queries_dict, docs, queries = PLSA.load(docs_path, queries_path)
    docs_keys   = list(docs_dict.keys())
    # PLSA.train(epochs)
    PLSA.test(queries)
    # with open('result.txt', 'w', encoding='utf-8') as fr:
    #     fr.write('Query,RetrievedDocuments\n')
    #     print('searching thought queries...')
    #     for index in tqdm(queries_dict):
    #         result = PLSA.search(queries_dict[index])
    #         fr.write('{}, '.format(index))
    #         for res in result:
    #             fr.write('{} '.format(docs_keys[res[0]]))
    #         fr.write('\n')


class config():
    def __init__(self):
        self.docs_path    = '../datasets/q_100_d_10000/docs'
        self.queries_path = '../datasets/q_100_d_10000/queries'
        self.epochs = 100

if __name__ == '__main__':
    cfg = config()
    experiment = wandb.init(project='IR_HW4_PLSA', resume='allow')
    experiment.config.update(vars(cfg))
    plsa(cfg.docs_path, cfg.queries_path, cfg.epochs, experiment)
    