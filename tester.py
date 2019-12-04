import sys, time
import numpy as np
from scipy import stats

from datatools import load_dataset
from classifier_bow import Classifier
# from eval import eval_file, eval_list, load_label_output

def set_reproducible():
    # The below is necessary to have reproducible behavior.
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(17)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)

def eval_list(glabels, slabels):
    if (len(glabels) != len(slabels)):
        print("\nWARNING: label count in system output (%d) is different from gold label count (%d)\n" % (
        len(slabels), len(glabels)))
    n = min(len(slabels), len(glabels))
    incorrect_count = 0
    for i in range(0, n):
        if slabels[i] != glabels[i]: incorrect_count += 1
    acc = (n - incorrect_count) / n
    acc = acc * 100
    return acc


def train_and_eval_dev_test(trainfile, devfile, testfile, run_id):
    classifier = Classifier()
    print("\n")
    # Training
    print("RUN: %s" % str(run_id))
    print("  %s.1. Training the classifier..." % str(run_id))
    classifier.train(trainfile, devfile)
    print()
    print("  %s.2. Evaluation on the dev dataset..." % str(run_id))
    slabels = classifier.predict(devfile)
    glabels = load_dataset(devfile)
    glabels = glabels['polarity']
    devacc = eval_list(glabels, slabels)
    print("       Acc.: %.2f" % devacc)
    testacc = -1
    if testfile is not None:
        # Evaluation on the test data
        print("  %s.3. Evaluation on the test dataset..." % str(run_id))
        slabels = classifier.predict(testfile)
        glabels = load_dataset(testfile)
        glabels = glabels['polarity']
        testacc = eval_list(glabels, slabels)
        print("       Acc.: %.2f" % testacc)
    print()
    return (devacc, testacc)

if __name__ == "__main__":
    set_reproducible()
    datadir = "../data/"
    trainfile =  datadir + "frdataset1_train.csv"
    devfile =  datadir + "frdataset1_dev.csv"
    # testfile =  datadir + "frdataset1_test.csv"
    testfile = None
    # Basic checking
    start_time = time.perf_counter()
    n = 5
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    devaccs = []
    testaccs = []
    for i in range(n):
        res = train_and_eval_dev_test(trainfile, devfile, testfile, i+1)
        devaccs.append(res[0])
        testaccs.append(res[1])
    print('\nCompleted %d runs.' % n)
    print("Dev accs:", devaccs)
    print("Test accs:", testaccs)
    print()
    print("Mean Dev Acc.: %.2f (%.2f)\tMean Test Acc.: %.2f (%.2f)" % (np.mean(devaccs), np.std(devaccs), np.mean(testaccs), np.std(testaccs)))
    total_exec_time = (time.perf_counter()-start_time)
    print("\nExec time: %.2f s. ( %d per run )" % (total_exec_time, total_exec_time/n))




