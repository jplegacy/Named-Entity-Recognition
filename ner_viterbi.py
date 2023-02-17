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
    pass


if __name__ == "__main__":
    print("\nLoading the data ...")
    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

    LABELS = ['B-LOC' 'B-MISC' 'B-ORG' 'B-PER' 'I-LOC' 'I-MISC' 'I-ORG' 'I-PER' 'O']

    print("\nTraining ...")
    train_feats = []
    train_labels = []

    # The vectorizer turns our features into vectors of numbers.
    vectorizer = DictVectorizer()
    classifier = LogisticRegression(max_iter=400)

    MeMM = MEMM(LABELS, [], vectorizer, classifier)

    for sent in train_sents:
        for i in range(len(sent)):
            for word_info in sent:
                MeMM.encode(word_info[0])

            feats = dict(word2features(sent, i))

            if i == 0:
                feats["0previousLabel"] = "<S>"
            else:
                feats["0previousLabel"] = train_labels[-1] # Grabs the last label made

            train_feats.append(feats)
            train_labels.append(sent[i][-1])


    X_train = MEMM.vectorizer.fit_transform(train_feats)
    # Not normalizing or scaling because the example feature is
    # binary, i.e. values are either 0 or 1.

    MeMM.classifier.fit(X_train, train_labels)

    print("\nTesting ...")
    # While developing use the dev_sents. In the very end, switch to
    # test_sents and run it one last time to produce the output file
    # results_memm.txt. That is the results_memm.txt you should hand
    # in.
    y_pred = []
    for sent in dev_sents:

        # sent_predi = model.predict(sent)



        # TODO: extract the feature representations for the words from
        # the sentence; use the viterbi algorithm to predict labels
        # for this sequence of words; add the result to y_pred
        pass

    print("Writing to results_memm.txt")
    # format is: word gold pred
    j = 0
    with open("results_memm.txt", "w") as out:
        for sent in dev_sents: 
            for i in range(len(sent)):
                word = sent[i][0]
                gold = sent[i][-1]
                pred = y_pred[j]
                j += 1
                out.write("{}\t{}\t{}\n".format(word,gold,pred))
        out.write("\n")

    print("Now run: python3 conlleval.py results_memm.txt")






