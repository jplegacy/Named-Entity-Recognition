import math

class MEMM:
    def __init__(self, states, vocabulary, vectorizer, classifier):
        """Save the components that define a Maximum Entropy Markov Model: set of
        states, vocabulary, and the classifier information.
        """
        self.states = states
        self.vocabulary = dict((vocabulary[i], i) for i in range(len(vocabulary)))
        self.vectorizer = vectorizer
        self.classifier =


    # TODO: Add additional methods that are needed. In particular, you
    # will need a method that can take a dictionary of features
    # representing a word and the tag chosen for the previous word and
    # return the probabilities of each of the MEMM's states.

    def encode(self, observation):
        """Returns the encoding of the given observation."""
        return self.vocabulary[observation]


def logprob(p):
    """Returns the logarithm of p."""
    if p != 0:
        return math.log(p)
    else:
        return float('-inf')

def rawprob(logprob):
    return pow(math.e, logprob)
