import math

class MEMM:
    def __init__(self, states, v, classifier):
        """Save the components that define a Maximum Entropy Markov Model: set of
        states, vocabulary, and the classifier information.
        """
        self.states = states
        self.vectorizer = v
        self.classifier = classifier

    def state_probability(self, feat_dict, previous_label):
        feat_dict["0previousLabel"] = previous_label
        feature_vec = self.vectorizer.fit_transform(feat_dict)
        return self.classifier.predict_proba(feature_vec)



    # TODO: Add additional methods that are needed. In particular, you
    # will need a method that can take a dictionary of features
    # representing a word and the tag chosen for the previous word and
    # return the probabilities of each of the MEMM's states.

