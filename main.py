# coding: utf8
import numpy as np
from rescal import *
import logging
import sys
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import time
import argparse
from numpy.linalg import norm
from commonFunctions import *

def setting_logger(log):
    # Create console handler
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s:%(name)s %(message)s")
    ch.setFormatter(formatter)
    log.addHandler(ch)

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
            # out_result.write("Score: %s\t%s\t%s: %f\n" % (e1, r, train.get_all_entity()[i], score))
            # out_result.flush()
            _log.debug("Score: %s\t%s\t%s: %f" % (e1, r, train.get_all_entity()[i], score))
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
    # out_result.write("Precision: %.3f Recall: %.3f\n" % ((hits * 1.0 / predicted), (hits * 1.0 / total)))
    # out_result.write("Hited: %d Predicted: %d Total: %d\n" % (hits, predicted, total) )
    out_result.write("%.3f %.3f\n" % ((hits * 1.0 / predicted), (hits * 1.0 / total)))
    out_result.write("%d %d %d\n" % (hits, predicted, total) )
    out_result.flush()
    _log.info("[Result] Precision: %.3f Recall: %.3f" % ((hits * 1.0 / predicted), (hits * 1.0 / total)))
    _log.info("[Result] Hited: %d Predicted: %d Total: %d" % (hits, predicted, total) )

def evaluate_relation_ROC(test, train, threshold):
    # Result: "[[relation, true_relation], [relation, true_relation], ...]"
    result = []

    total = test.get_num_triple()

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
            _log.debug("Score: %s\t%s\t%s: %f" % (e1, train.get_all_relation()[i], e2.strip(), score))
            if max_score < score:
                max_score = score
                max_idx = i
        result.append([r, train.get_all_relation()[max_idx]])
    return result

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
            _log.debug("Score: %s\t%s\t%s: %f" % (e1, train.get_all_relation()[i], e2.strip(), score))
            if max_score < score:
                max_score = score
                max_idx = i

        if max_score > threshold:
            predicted += 1
            if actual_relation == max_idx:
                hits += 1
    if predicted == 0:
        return 0
    out_result.write("Precision: %.3f Recall: %.3f\n" % ((hits * 1.0 / predicted), (hits * 1.0 / total)))
    out_result.write("Hited: %d Predicted: %d Total: %d\n" % (hits, predicted, total) )
    out_result.flush()
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
                _log.debug("Score: %s\t%s\t%s: %f" % (e1, train.get_all_relation()[k], e2.strip(), score))
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
    out_result.write("Precision: %.3f Recall: %.3f\n" % ((hits * 1.0 / n_precision), (hits * 1.0 / n_recall)))
    out_result.write("Hited: %d Predicted: %d Total: %d\n" % (hits, n_precision, n_recall) )
    out_result.flush()
    _log.info("[Result] Precision: %.3f Recall: %.3f" % ((hits * 1.0 / n_precision), (hits * 1.0 / n_recall)))
    _log.info("[Result] Hited: %d Predicted: %d Total: %d" % (hits, n_precision, n_recall) )


def in_test(triple, test):
    for e1, r, e2 in test.get_triple():
        if triple[0] == e1 and triple[2] == e2:
            return True, r
    return False, None

def roc(evaluation):
    allRel = []
    totalScore = 0
    totalAccuracy = 0
    for e in evaluation:
        allRel.append(e[0])
        allRel.append(e[1])
    allRel = list(set(allRel))

    for i, rel in enumerate(allRel):
        true = []
        score = []
        for sample in evaluation:
            true.append(1 if sample[0] == rel else 0)
            score.append(1 if sample[1] == rel else 0)
        totalScore += roc_auc_score(np.array(true), np.array(score))
        totalAccuracy += accuracy_score(np.array(true), np.array(score))

    count = 0
    for sample in evaluation:
        if sample[0] == sample[1]:
            count += 1
    print "Count:", count, " Evaluation: ", len(evaluation)
    return totalScore * 1.0 / len(allRel), totalAccuracy * 1.0 / len(allRel)

if __name__ == '__main__':
    # Parsing arguments

    start_time = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="train file", required=True)
    parser.add_argument("--test", type=str, help="test file", required=True)
    parser.add_argument("--latent", type=int, help="number of latent components", default=2, required=False)
    parser.add_argument("--lmbda", type=float, help="regularization parameter", default=0, required=False)
    parser.add_argument("--th", type=float, help="regularization parameter", default=0, required=False)
    parser.add_argument("--log", type=str, help="log file", default="rescal.log", required=False)
    parser.add_argument("--result", type=str, help="result file", default="result.txt",  required=False)
    # parser.add_argument("--outputentities", type=str, help="the file, where the latent embedding for entities will be output", required=True)
    # parser.add_argument("--outputfactors", type=str, help="the file, where the latent factors will be output", required=True)

    args = parser.parse_args()
    logFile = args.log
    logFile  = "log/" + logFile
    result_file = args.result
    result_file = "result/" + result_file
    numLatentComponents = args.latent
    regularizationParam = args.lmbda
    train_file = args.train
    test_file = args.test
    th = args.th
    # outputEntities = args.outputentities
    # outputFactors = args.outputfactors

    # Initialize logger
    out_result = open(result_file, "w")
    _log = logging.getLogger('RESCAL') 
    setting_logger(_log)
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO, format="%(asctime)s:%(name)s %(message)s")
    # _log.setLevel(logging.INFO)

    # Load train and test dataset
    ds_train = dataset(train_file, "UTF-8")
    ds_test = dataset(test_file, "UTF-8")
    ds_train.load_triples()
    ds_test.load_triples()

    _log.info("-" * 50)
    _log.info("[Dataset] Trainset: " + repr(ds_train))
    _log.info("[Dataset] Testset: " + repr(ds_test))
    _log.info("-" * 50)

    # Buils sparse matrix
    X = ds_train.build_csr_matrix()
    _log.info('[Dataset] The number of entities: %d' % ds_train.get_num_entity())
    _log.info('[Dataset] The number of relations: %d' % ds_train.get_num_relation())
    _log.info("-" * 50)

    # for numLatentComponents in (100, 150, 200):
        # for t in [i * 0.1 for i in range(10)]:
    for t in (0, ):
        result = rescal(X, numLatentComponents, lmbda=regularizationParam)
        _log.info('[Tensor] Objective function value: %.3f' % result[2])
        _log.info('[Tensor] Of iterations: %d' % result[3])

        A = result[0]
        R = result[1]
        _log.info("[Tensor] Matrix A's shape: %s" % str(A.shape))
        _log.info("[Tensor] Matrix R's shape: %s" % str(R[0].shape))
        # _log.info("## Execute time: %s" % str(sum(result[4])))
        # Evaluate algorithm performance
        out_result.write("-" * 20 + "\n")
        # out_result.write("Latent: %d, Threshold: %f\n" % (numLatentComponents, t))
        out_result.write("%d %.1f\n" % (numLatentComponents, t))
        out_result.flush()
        _log.info("[Iterator] Latent: %d, Threshold: %.1f" % (numLatentComponents, t))
        # evaluate_relation_2(ds_test, ds_train, t)
        evaluation = evaluate_relation_ROC(ds_test, ds_train, t)
        r, p = roc(evaluation)
        print r, p

    end_time = datetime.now()
    out_result.close()
    _log.info("Totally finished in %ds" % ((end_time - start_time).seconds))
    # _log.info("Totally finished in %.3fs" % ((end_time - start_time).microseconds * 1.0 / 1000))
