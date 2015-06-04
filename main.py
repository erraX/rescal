# coding: utf8
import argparse
import logging
import numpy as np
import sys
import time
from logger import Logger
from commonFunctions import *
from datetime import datetime
from numpy.linalg import norm
from dataset import dataset
from rescal import *
from eval import *
from util import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

class RunRescal:
    def __init__(self, train, test, logger):
        self.train = train
        self.test = test
        self.train.load_triples()
        self.test.load_triples()
        self.X = self.train.build_csr_matrix()
        self.logger = logger
        self.A = None
        self.R = None
        self.scores = None

    def rescal(self, config):
        factorization = rescal(self.X, config['numLatentComponents'], lmbda=config['regularizationParam'])
        self.A = factorization[0]
        self.R = factorization[1]

    # 1. 是否设定阈值
    # 2. 
    def predict(self):
        pass

    def calEntityPairScore(self, enIdx1, enIdx2):
        '''
        Return score:
            [('father': 0.9), ('mother': 0.5), ...]
        '''
        score = []
        for idxR in range(len(self.R)):
            # score[self.R[idxR]] = np.dot(self.A[enIdx1,:], np.dot(self.R[idxR], self.A.T[:, enIdx2]))
            score.append((self.R[idxR], np.dot(self.A[enIdx1,:], np.dot(self.R[idxR], self.A.T[:, enIdx2]))))
        return score
   

    def calEveryRelationScore(self):
        '''
        Return scores:
            {
                1:[('father': 0.9), ('mother': 0.5), ...],
                2:[('son': 0.8), ('brother': 0.2), ...],
                3:[],  因为3没有出现在训练集中,所以没有score
                ...
            }
        '''
        scores = {}
        for idx, triple in enumerate(self.test.get_triple()):
            e1, r, e2 = triple[0:3]
            try:
                idx_e1 = self.train.get_idx_entity()[e1]
                idx_e2 = self.train.get_idx_entity()[e2]
            except:
                self.logger.error("在训练集中没有找到实体或关系: %" % ','.join(triple))
                score[idx] = {}
                continue
            score[idx] = calEntityPairScore(idx_e1, idx_e2)
        self.scores = scores
        return scores

    def pickPredictedResult(self, threshold=0):
        '''
        Return testCase:
            [['en1', 'rel1', 'en2', 'prerel1'], ...]
        '''
        testCase = self.test.get_triple()
        for idx, scoreHash in self.scores:
            if scoreHash[0][1] > threshold:
                # [('father', 0.9), ...]  取出得分最高的关系名
                testCase[idx] += scoreHash[0][0]
        return testCase

    # ROC, AUC, PRECISION, RECALL
    def evaluateEntity(self, method, config, verbose=False):
        pass

    def evaluateRelation(self, method, config, verbose=False):
        pass

    def evaluate_relation_ROC(self, threshold):
        # Result: "[[relation, true_relation], [relation, true_relation], ...]"
        result = []
        total = self.test.get_num_triple()
        for triple in self.test.get_triple():
            e1, r, e2 = triple[0:3]
            try:
                idx_e1 = self.train.get_idx_entity()[e1]
                idx_e2 = self.train.get_idx_entity()[e2]
                actual_relation = self.train.get_idx_relation()[r.strip()]
            except:
                continue
            max_score = 0
            max_idx = 0
            scores = []
            for i in range(len(R)):
                try:
                    score = np.dot(A[idx_e1,:], np.dot(R[i], A.T[:, idx_e2]))
                except:
                    score = 0
                scores.append(score)
            # scores = normalize(scores)
            for i, score in enumerate(scores):
                self.logger.getLog().debug("Score: %s\t%s\t%s: %f" % (e1, self.train.get_all_relation()[i], e2.strip(), score))
                if max_score < score:
                    max_score = score
                    max_idx = i
            result.append([r, self.train.get_all_relation()[max_idx]])
        return result

def evaluate_entity(test, train, threshold):
    predicted = 0
    total = test.get_num_triple()
    hits = 0

    for triple in test.get_triple():
        e1, r, e2 = triple[0:3]
        idx_e1 = train.get_idx_entity()[e1]
        actual_idx_e2 = train.get_idx_entity()[e2]
        idx_r = train.get_idx_relation()[r.strip()]
        max_score = 0
        max_idx = 0
        scores = []

        for i in range(train.get_num_entity()):
            score = np.dot(A[idx_e1,:], np.dot(R[idx_r], A.T[:, i]))
            scores.append(score)

        # scores = normalize(scores)
        # print scores
        for i, score in enumerate(scores):
            # resultToFile.write("Score: %s\0%s\t%s: %f\n" % (e1, r, train.get_all_entity()[i], score))
            # resultToFile.flush()
            _log.debug("Score: %s\0%s\t%s: %f" % (e1, r, train.get_all_entity()[i], score))
            if max_score < score:
                max_score = score
                max_idx = i
        _log.debug("-" * 50)

        if max_score > threshold:
            predicted += 1
            if actual_idx_e2 == max_idx:
                hits += 1

    if predicted == 0:
        return 0
    # resultToFile.write("Precision: %.3f Recall: %.3f\n" % ((hits * 1.0 / predicted), (hits * 1.0 / total)))
    # resultToFile.write("Hited: %d Predicted: %d Total: %d\n" % (hits, predicted, total) )
    resultToFile.write("%.3f %.3f\n" % ((hits * 1.0 / predicted), (hits * 1.0 / total)))
    resultToFile.write("%d %d %d\n" % (hits, predicted, total) )
    resultToFile.flush()
    _log.info("[Result] Precision: %.3f Recall: %.3f" % ((hits * 1.0 / predicted), (hits * 1.0 / total)))
    _log.info("[Result] Hited: %d Predicted: %d Total: %d" % (hits, predicted, total) )

def evaluate_relation(test, train, threshold):
    # 是正类被预测成正类
    TP = []
    # 是负类被预测成正类
    FP = []
    # 是负类被预测成负类
    TN = []
    # 是正类被预测成负类
    FN = []
    predicted = 0
    total = test.get_num_triple()
    hits = 0

    for triple in test.get_triple():
        e1, r, e2 = triple[0:3]
        try: 
            idx_e1 = train.get_idx_entity()[e1]
            idx_e2 = train.get_idx_entity()[e2]
            actual_relation = train.get_idx_relation()[r.strip()]
        except:
            continue
        max_score = 0
        max_idx = 0
        scores = []

        for i in range(len(R)):
            try:
                score = np.dot(A[idx_e1,:], np.dot(R[i], A.T[:, idx_e2]))
            except:
                score = 0
            scores.append(score)

        # scores = normalize(scores)
        for i, score in enumerate(scores):
            _log.debug("Score: %s\0%s\t%s: %f" % (e1, train.get_all_relation()[i], e2.strip(), score))
            if max_score < score:
                max_score = score
                max_idx = i

        if max_score > threshold:
            predicted += 1
            if actual_relation == max_idx:
                hits += 1
    if predicted == 0:
        return 0
    resultToFile.write("Precision: %.3f Recall: %.3f\n" % ((hits * 1.0 / predicted), (hits * 1.0 / total)))
    resultToFile.write("Hited: %d Predicted: %d Total: %d\n" % (hits, predicted, total) )
    resultToFile.flush()
    _log.info("[Result] Precision: %.3f Recall: %.3f" % ((hits * 1.0 / predicted), (hits * 1.0 / total)))
    _log.info("[Result] Hited: %d Predicted: %d Total: %d" % (hits, predicted, total) )


def evaluate_relation_2(test, train, threshold):
    hits = 0
    n_precision = 0
    n_recall = test.get_num_triple()

    for i in range(train.get_num_entity()):
        print "i: %d" %i
        for j in range(train.get_num_entity()):
            e1 = train.get_all_entity()[i]
            e2 = train.get_all_entity()[j]

            max_score = 0
            max_idx = 0
            scores = []

            for k in range(len(R)):
                try:
                    score = np.dot(A[i,:], np.dot(R[k], A.T[:, j]))
                except:
                    score = 0
                scores.append(score)

            # scores = normalize(scores)
            for k, score in enumerate(scores):
                _log.debug("Score: %s\0%s\t%s: %f" % (e1, train.get_all_relation()[k], e2.strip(), score))
                if max_score < score:
                    max_score = score
                    max_idx = k

            if max_score > threshold:
                n_precision += 1
                relation = train.get_all_relation()[max_idx]
                is_in, r = in_test((e1, relation, e2), test)
                if is_in and r == relation:
                    hits += 1
    if n_precision == 0:
        return 0
    resultToFile.write("Precision: %.3f Recall: %.3f\n" % ((hits * 1.0 / n_precision), (hits * 1.0 / n_recall)))
    resultToFile.write("Hited: %d Predicted: %d Total: %d\n" % (hits, n_precision, n_recall) )
    resultToFile.flush()
    _log.info("[Result] Precision: %.3f Recall: %.3f" % ((hits * 1.0 / n_precision), (hits * 1.0 / n_recall)))
    _log.info("[Result] Hited: %d Predicted: %d Total: %d" % (hits, n_precision, n_recall) )

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="train file", required=True)
    parser.add_argument("--test", type=str, help="test file", required=True)
    parser.add_argument("--latent", type=int, help="number of latent components", default=2, required=False)
    parser.add_argument("--lmbda", type=float, help="regularization parameter", default=0, required=False)
    parser.add_argument("--th", type=float, help="regularization parameter", default=0, required=False)
    parser.add_argument("--log", type=str, help="log file", default="rescal.log", required=False)
    parser.add_argument("--result", type=str, help="result file", default="result.txt",  required=False)
    return parser.parse_args()

if __name__ == '__main__':
    # Parsing arguments
    start_time = datetime.now()
    cliInputArgs = parseArguments()

    # Log file
    logFile = "./log/" + cliInputArgs.log
    # Result file
    resultToFile = open("./result/" + cliInputArgs.result, 'w')

    logger = Logger()
    runRescal = RunRescal(dataset(cliInputArgs.train, "UTF-8"), dataset(cliInputArgs.test, "UTF-8"), logger)
    config = {'numLatentComponents':cliInputArgs.latent, 'regularizationParam':cliInputArgs.lmbda, 'th':cliInputArgs.th}
    runRescal.rescal(config)

    # Start training
    result = rescal(runRescal.X, config['numLatentComponents'], lmbda=config['regularizationParam'])
    logger.getLog().info('[Tensor] Objective function value: %.3f' % result[2])
    logger.getLog().info('[Tensor] Of iterations: %d' % result[3])

    A = result[0]
    R = result[1]
    logger.getLog().info("[Tensor] Matrix A's shape: %s" % str(A.shape))
    logger.getLog().info("[Tensor] Matrix R's shape: %s" % str(R[0].shape))
    # _log.info("## Execute time: %s" % str(sum(result[4])))
    # Evaluate algorithm performance
    resultToFile.write("-" * 20 + "\n")
    # resultToFile.write("Latent: %d, Threshold: %f\n" % (numLatentComponents, 0))
    resultToFile.write("%d %.1f\n" % (config['numLatentComponents'], 0))
    resultToFile.flush()
    logger.getLog().info("[Iterator] Latent: %d, Threshold: %.1f" % (config['numLatentComponents'], 0))
    # evaluate_relation_2(ds_test, ds_train, 0)
    evaluation = runRescal.evaluate_relation_ROC(0)
    r, p = roc(evaluation)

    end_time = datetime.now()
    resultToFile.close()
    logger.getLog().info("Totally finished in %ds" % ((end_time - start_time).seconds))
