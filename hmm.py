###
### Author: Kristina Striegnitz
### Version: Winter 2023
###
### Definition of a Hidden Markov Model
###
class HMM:
    def __init__(self, states, vocabulary, start_probs, trans_probs, emit_probs):
        """Save all components that make up a Hidden Markov Model: set of
        states, vocabulary, start probabilities, transition probabilities,
        emission probabilities.
        """
        self.states = states
        self.vocabulary = dict((vocabulary[i], i) for i in range(len(vocabulary)))
        self.start_probs = start_probs
        self.trans_probs = trans_probs
        self.emit_probs = emit_probs

    def start(self, state):
        """Returns the start probability for a given state."""
        return self.start_probs[state]

    def emit(self, state, observation):
        """Returns the emission probability for a given state and
           observation."""
        return self.emit_probs[state][observation]

    def transition(self, from_state, to_state):
        """Returns the probabitility to transition from one state to
           another."""
        return self.trans_probs[from_state][to_state]

    def encode(self, observation):
        """Returns the encoding of the given observation."""
        return self.vocabulary[observation]
