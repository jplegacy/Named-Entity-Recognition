import math

class MEMM:
    def __init__(self, states, v, classifier):
        """Save the components that define a Maximum Entropy Markov Model: set of
        states, vocabulary, and the classifier information.
        """
        self.states = states
        self.vectorizer = v
        self.classifier = classifier

    def state_probabilities(self, window_dict, previous_label):
        window_dict["0previousWord"] = previous_label
        feature_vec = self.vectorizer.transform(window_dict)
        return self.classifier.predict_log_proba(feature_vec)

    # def get_state_probability(self, list_of_prob, wanted_state):
    #     desired_index = 0
    #     for i, state in enumerate(self.states):
    #         if wanted_state == state:
    #             desired_index = i
    #     return list_of_prob[0][desired_index]
