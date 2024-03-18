from lambeq.backend.grammar import Cup, Diagram, Id, Swap, Ty, Word
from lambeq.backend.drawing import draw, draw_equation
from lambeq import IQPAnsatz, RemoveCupsRewriter

## define the grammatical types
# note the "h" type for honorifics/polite form
n, s, h = Ty('n'), Ty('s'), Ty ('h') 

## define utilities and hyperparameters
BATCH_SIZE = 10
EPOCHS = 100
SEED = 2
remove_cups = RemoveCupsRewriter()
ansatz = IQPAnsatz({n: 1, s: 1, h: 1}, n_layers=1, n_single_qubit_params=3)

## define grammar model
# polite
pred_polite_n = n @ h.l
verb_connective_single = s.l @ s @ h.l     
verb_connective_double = s.l @ s.l @ s @ h.l
verb_connective_triple = s.l @ s.l @ s.l @ s @ h.l
adj_i_s_p = verb_connective_single
masu = h
desu_i_adj = masu
desu_n = h @ n.r @ s.l @ s 
title = h
name = n @ h.l               
# casual
verb_casual_single = s.l @ s
verb_casual_double = s.l @ s.l @ s
verb_casual_triple = s.l @ s.l @ s.l @ s
te_form = s.l @ s.l.l
da = n.r @ s.l @ s 
adj_i_s = verb_casual_single   
adj_i = n @ n.l
adj_w_na = adj_i
adj_wo_na = n 
adv = s.l @ s.l.l    
pro = n
cas_name = n
# particles (ha, ga, ni, de, wo in single sentence context)
particle_logical = n.r @ s.l.l

## Open datasets and collect sentences
def read_data(filename):
    labels, sentences = [], []
    with open(filename) as file:
        for line in file:
            t = int(line[0])
            labels.append([t, 1-t]) # returns an array with the correct label in first place
            sentences.append(line[1:].strip())
    return labels, sentences

## Diagramizer Algorithm
def diagramizer(words):
    return ""

## Procedure
# read in data and labels
train_labels, train_data = read_data('./datasets/shuffled/trainsetshuff.txt')
val_labels, val_data = read_data('./datasets/shuffled/valsetshuff.txt')
test_labels, test_data = read_data('./datasets/shuffled/testsetshuff.txt')
# test file reading code
print(train_data[:5])
print(train_labels[:5])

# Use diagramizer to build diagrams
