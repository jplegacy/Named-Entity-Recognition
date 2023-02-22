import math

class MEMM:
    def __init__(self, v, classifier):
        """Save the components that define a Maximum Entropy Markov Model: set of
        states, vocabulary, and the classifier information.
        """
        self.vectorizer = v
        self.classifier = classifier

    def state_probability(self, window_dict, previous_label):
        window_dict["0previousWord"] = previous_label
        feature_vec = self.vectorizer.fit_transform(window_dict)
        print(feature_vec)

        return self.classifier.predict_log_proba(feature_vec)

    def states(self):
        return self.classifier.classes_