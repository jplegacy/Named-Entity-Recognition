"""Named Entity Recognition as a sequence tagging task.

Author: Kristina Striegnitz and <YOUR NAME HERE>

<HONOR CODE STATEMENT HERE>

Complete this file for part 2 of the project.
"""
from nltk.corpus import conll2002
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

import math
import numpy as np
from memm import MEMM

#################################
#
# Word classifier
#
#################################

def getfeats(word, o):
    """Take a word its offset with respect to the word we are trying to
    classify. Return a list of tuples of the form (feature_name,
    feature_value).
    """
    o = str(o)
    features = [
        (o + 'word', word),
        (o + 'isUpper', word[0].isupper()),
        (o + 'isAlpha', word.isalpha()),
        (o + 'isTitle', word.istitle())
    ]
    return features
    

def word2features(sent, i):
    """Generate all features for the word at position i in the
    sentence. The features are based on the word itself as well as
    neighboring words.
    """
    features = []
    # the window around the token
    for o in [-1,0,1]:
        if i+o >= 0 and i+o < len(sent):
            word = sent[i+o][0]
            featlist = getfeats(word, o)
            features.extend(featlist)
    return features


#################################
#
# Viterbi decoding
#
#################################
def viterbi(obs, memm, pretty_print=False):
    # OBS - A list of dictionaries in a sentence

    V = [{}]
    path = {}

    initial_state_probabilities = memm.state_probabilities(obs[0], "<s>")

    for y in memm.states:
        V[0][y] = memm.get_state_probability(initial_state_probabilities, y)
        path[y] = [y]

    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in memm.states :
            max_v = float('-inf')
            max_prev_state = None

            for prev_y in memm.states:
                v = V[t - 1][prev_y] + memm.get_state_probability(initial_state_probabilities, prev_y)

                if v > max_v:
                    max_v = v
                    max_prev_state = prev_y
            V[t][y] = max_v
            newpath[y] = path[max_prev_state] + [y]

        # Don't need to remember the old paths
        path = newpath

    (prob, state) = max([(V[len(obs) - 1][y], y) for y in memm.states])
    return path[state]

if __name__ == "__main__":
    print("\nLoading the data ...")
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    print("\nTraining ...")
    train_feats = []
    train_labels = []

    for sent in train_sents:
        for i in range(len(sent)):
            word_feat = dict(word2features(sent, i))
            if i == 0:
                word_feat["previousLabel"] = "<S>"
            else:
                word_feat["previousLabel"] = sent[i-1][-1]

            train_feats.append(word_feat)
            train_labels.append(sent[i][-1])

    vectorizer = DictVectorizer()
    # The vectorizer turns our features into vectors of numbers.
    X_train = vectorizer.fit_transform(train_feats)

    # Not normalizing or scaling because the example feature is
    # binary, i.e. values are either 0 or 1.
    classifier = LogisticRegression(max_iter=400)
    classifier.fit(X_train, train_labels)

    memm = MEMM(classifier.classes_, vectorizer, classifier)

    print("\nTesting ...")
    # While developing use the dev_sents. In the very end, switch to
    # test_sents and run it one last ti me to produce the output file
    # results_memm.txt. That is the results_memm.txt you should hand
    # in.
    y_pred = []
    for sent in dev_sents:
        sent_feats = []
        for i in range(len(sent)):
            feats = dict(word2features(sent, i))
            sent_feats.append(feats)
        y_pred.append(viterbi(sent_feats, memm))

    print("Writing to results_memm.txt")
    # format is: word gold pred
    j = 0
    with open("results_memm.txt", "w") as out:
        for sent in dev_sents:
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j][i]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python3 conlleval.py results_memm.txt")






