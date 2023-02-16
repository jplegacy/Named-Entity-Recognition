###
### Author: Kristina Striegnitz
### Version: Winter 2023
###
### An implementation of Viterbi decoding for the ice cream HMM
### example using numpy matrix operations.
###
import math
import numpy as np
from hmm import HMM

def viterbi(obs, hmm, pretty_print=False):
    """The Viterbi algorithm. Calculates what sequence of states is most
    likely to produce the given sequence of observations.

    V is the main data structure that represents the trellis/grid of
    Viterbi probabilities. It is a list of np.arrays. Each np.array
    represents one column in the grid.

    The variable 'path' maintains the currently most likely paths. For
    example, once we have finished processing the third observation,
    then 'path' contains for each possible state, the currently most
    likely path that leads to that state given these three
    observations.
    """
    V = []
 
    # Calculate the Viterbi probabilities for the first step, i.e.,
    # the first observation o0.
    o0 = hmm.encode(obs[0])
    V.append(np.log(hmm.start_probs)+np.log(hmm.emit_probs[..., o0]))
    paths =  np.array([[i] for i in range(len(hmm.states))])
    
    # Run Viterbi for all of the subsequent steps/observations: t > 0.
    for t in range(1,len(obs)):
        ot = hmm.encode(obs[t])
        transition_probs = V[t-1] + np.log(hmm.trans_probs.T)
        emission_prob = np.log(hmm.emit_probs[..., ot])
        V.append(np.max(transition_probs, axis=1) + emission_prob)
        max_prev_states = np.argmax(transition_probs, axis=1)
        paths = np.insert(paths[max_prev_states], t, range(len(hmm.states)), axis=1)
        
    if pretty_print:
        pretty_print_trellis(V)

    most_probable_final_state = np.argmax(V[len(obs)-1])
    most_probable_path = [hmm.states[i] for i in paths[most_probable_final_state]]
    return most_probable_path


def pretty_print_trellis(V):
    """Prints out the Viterbi trellis to the screen formatted as a grid."""
    print("    ", end=" ")
    for i in range(len(V)):
        print("%7s" % ("o %d" % i), end=" ")
    print()
 
    for j in range(len(V[0])):
        print("s %.5s: " % j, end=" ")
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][j]), end=" ")
        print()


###
## Example HMM specification
###

def example_hmm():
    states = ['Hot', 'Cold']
    vocabulary = ['1', '2', '3']
    # start_probabilities = {'Hot': 0.8, 'Cold': 0.2}
    start_probabilities = np.array([0.8, 0.2])
    # transition_probabilities = {
    #     'Hot' : {'Hot': 0.7, 'Cold': 0.3},
    #     'Cold' : {'Hot': 0.4, 'Cold': 0.6}
    #     }
    transition_probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
    # emission_probabilities = {
    #     'Hot' : {'1': 0.2, '2': 0.4, '3': 0.4},
    #     'Cold' : {'1': 0.5, '2': 0.4, '3': 0.1}
    #     }
    emission_probabilities = np.array([[0.2, 0.4, 0.4], [0.5, 0.4, 0.1]])
    ice_cream_hmm = HMM(states, vocabulary, start_probabilities, transition_probabilities, emission_probabilities)
    return ice_cream_hmm


###
# Run example
###
def run_example(observations, hmm ,pretty_print=True):
    """Run Viterbi on example observations using given HMM."""
    return viterbi(observations, hmm, pretty_print)

if __name__ == '__main__':    
    hmm = example_hmm()
    obs = ('3', '1', '3')
    print("Observations:", obs)
    print()
    most_likely_state_sequence = run_example(obs, hmm)
    print()
    print("Most likely state sequence:", most_likely_state_sequence)

