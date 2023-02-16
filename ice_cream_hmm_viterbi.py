###
### Author: Kristina Striegnitz
### Version: Winter 2023
###
### An implementation of Viterbi decoding for the ice cream HMM
### example.
###
import math
from hmm import HMM

def viterbi(obs, hmm, pretty_print=False):
    """The Viterbi algorithm. Calculates what sequence of states is most
    likely to produce the given sequence of observations.

    V is the main data structure that represents the trellis/grid of
    Viterbi probabilities. It is a list of dictionaries. Each
    dictionary represents one column.

    The variable 'path' maintains the currently most likely path. For
    example, once we have finished processing the third observation,
    then 'path' contains for each possible state, the currently most
    likely path that leads to that state. 'path' is a dictionary with
    one entry for each possible state. And each value is a list of
    states, representing the most likely sequence of states leading to
    the state represented by the key.
    """
    V = [{}]
    path = {}
 
    # Calculate the Viterbi probabilities for the first step, i.e.,
    # the first observation: t = 0.
    for y in hmm.states:
        V[0][y] = logprob(hmm.start_probs[y]) + logprob(hmm.emit_probs[y][obs[0]])
        path[y] = [y]
 
    # Run Viterbi for all of the subsequent steps/observations: t > 0.
    for t in range(1,len(obs)):
        V.append({})
        newpath = {}

        for y in hmm.states:
            max_v = float('-inf')
            max_prev_state = None
            for prev_y in hmm.states:
                transition_prob = V[t-1][prev_y] + logprob(hmm.trans_probs[prev_y][y])
                emission_prob = logprob(hmm.emit_probs[y][obs[t]])
                v = transition_prob + emission_prob
                if v > max_v:
                    max_v = v
                    max_prev_state = prev_y
            V[t][y] = max_v
            newpath[y] = path[max_prev_state] + [y]
 
        # Don't need to remember the old paths
        path = newpath

    if pretty_print:
        pretty_print_trellis(V)
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in hmm.states])
    return path[state]


def pretty_print_trellis(V):
    """Prints out the Viterbi trellis formatted as a grid."""
    print("    ", end=" ")
    for i in range(len(V)):
        print("%7s" % ("%d" % i), end=" ")
    print()
 
    for y in V[0].keys():
        print("%.5s: " % y, end=" ")
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][y]), end=" ")
        print()

def logprob(p):
    """Returns the logarithm of p."""
    if p != 0:
        return math.log(p)
    else:
        return float('-inf')

def rawprob(logprob):
    return pow(math.e, logprob)

    
###
## Example HMM specification
###

def example_hmm():
    states = ['Hot', 'Cold']
    vocabulary = ['1', '2', '3']
    start_probabilities = {'Hot': 0.8, 'Cold': 0.2}
    transition_probabilities = {
        'Hot' : {'Hot': 0.7, 'Cold': 0.3},
        'Cold' : {'Hot': 0.4, 'Cold': 0.6}
    }
    emission_probabilities = {
        'Hot' : {'1': 0.2, '2': 0.4, '3': 0.4},
        'Cold' : {'1': 0.5, '2': 0.4, '3': 0.1}
    }
    ice_cream_hmm = HMM(states, vocabulary, start_probabilities, transition_probabilities, emission_probabilities)
    return ice_cream_hmm

###
# Run example
###
def run_example(observations, hmm, pretty_print=True):
    """Run Viterbi on example observations using given HMM."""
    return viterbi(observations, hmm, pretty_print)

if __name__=='__main__':
    hmm = example_hmm()
    obs = ('3', '1', '3')
    print("Observations:", obs)
    print()
    most_likely_state_sequence = run_example(obs, hmm)
    print()
    print("Most likely state sequence:", most_likely_state_sequence)
