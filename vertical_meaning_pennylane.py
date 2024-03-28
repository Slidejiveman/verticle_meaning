from lambeq.backend.grammar import Cup, Diagram, Ty, Word
from lambeq import AtomicType, Dataset, IQPAnsatz, PennyLaneModel, PytorchTrainer, RemoveCupsRewriter, UnifyCodomainRewriter
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import random
import torch

## define the grammatical types
# note the "h" type for honorifics/polite form
n, s, h = AtomicType.NOUN, AtomicType.SENTENCE, Ty ('h') 

## define utilities and hyperparameters
BATCH_SIZE = 2
EPOCHS = 15
LEARNING_RATE = 0.1
SEED = 42
optimizer = torch.optim.AdamW #figure out how to save and load the best model in the hybrid case
log_dir = "./logs/hybrid"
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
remove_cups = RemoveCupsRewriter()
u_cdom_rewriter = UnifyCodomainRewriter(s)
ansatz = IQPAnsatz({n: 1, s: 1, h: 1}, n_layers=2, n_single_qubit_params=3)

## define grammar model. NOTE: There are extra types defined here to capture possible avenues of research
# polite
pred_polite_n = n @ h.l
verb_connective = s @ h.l # particle_links @ verb connective
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
verb_casual = s
verb_casual_single = s.l @ s
verb_casual_double = s.l @ s.l @ s
verb_casual_triple = s.l @ s.l @ s.l @ s
te_form = s.l @ s.l.l
da = n.r @ s.l @ s 
adj_i_s = verb_casual_single   
adj_i = n @ n.l
adj_w_na = adj_i
adj_wo_na = n 
adv = s.l.l    
pro = n
cas_name = n
# particles (ha, ga, ni, de, wo in single sentence context)
particle_logical = n.r @ s.l.l
possessive_no = n.r @ n.l
with_to = particle_logical
and_to_ya_toka = possessive_no

## Define special words
particles = ["ha", "ga", "ni", "de", "wo"]
titles = ["san"]
honorific_words = ["desu", "masu"] + titles
casual_copula = ["da"]
possessives = ["no"]
special_words = casual_copula + honorific_words + particles + possessives

## Define loss and accuracy functions
def acc(y_hat, y):
    return (torch.argmax(y_hat, dim=1) ==
            torch.argmax(y, dim=1)).sum().item()/len(y)

def loss(y_hat, y):
    return torch.nn.functional.mse_loss(y_hat, y)

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
# Receives a sentence with tokens separated by spaces.
# Passed tests: "kanojo ga kaeru", "kanojo ga kaeri masu", "kanojo ga kami wo ori masu", 
# "kanojo ga tsukue de kami wo ori masu", "kanojo ga tsukue de kami wo oru", "kanojo ga utsukushii"
# "hon ga utsukushii desu", "Suzuki san ga utsukushii", "Suzuki san ga Tanaka san wo hanasu"
# "kanojo ga kami wo utsukushiku oru", "kanojo ga booru wo motte ki masu", "kanojo ga booru wo motte kuru",
# "hon ga kantan da", "Suzuki san ga kirei desu", and ""
def diagramizer(sentence):    
    # Put the input sentence into an ordered list of words
    if sentence is not None and sentence != "":
        words = sentence.split()
        print(words)
    else:
        print("Diagramizer's input cannot be empty.")
        return
    
    # Count the number of input words
    word_count = len(words)
    print(f"word count: {word_count}")
    
    # Determine positions of special words to aid in detecting the presence of adjectives or adverbs
    special_word_indices = {special_word: [] for special_word in special_words}
    for index, special_word in enumerate(words):
        if special_word in special_word_indices:
            special_word_indices[special_word].append(index)
    print("Indices and counts of special words: ")
    for special_word, special_word_index_list in special_word_indices.items():
        print(f"{special_word}: {special_word_index_list}, Count: {len(special_word_index_list)}")
    
    ## Assign types to substrings to turn them into a list of Word types
    # First: Determine the number of particles present
    particle_count = sum(len(special_word_indices[particle]) for particle in particles if particle in special_word_indices)
    print(f"particle count: {particle_count}")
    # assuming a hidden "ga" so that there is always at least one particle.
    particle_links = s.l
    for _ in range(1, particle_count):
        particle_links = particle_links @ s.l
    print(f"particle_links value: {particle_links}")
    
    # Second: Check ending word for "desu", "masu", "da", or a word that ends in "i".
    sentence_ender = None
    meaning_carrier = None
    adv_index = None
    te_index = None
    noun_verb_index = None
    if special_word_indices["desu"]:
        print("desu found.")
        # it is possible that the following check will catch a name like "Genji".
        # for now, this assumption is safe enough to work with. Filtering for common names could improve this.
        if words[-2][-1] == "i" and words[-2][-2:] != "ei":
            sentence_ender = desu_i_adj
            meaning_carrier = particle_links @ s @ h.l
        else:
            sentence_ender = h @ n.r @ particle_links @ s
    elif special_word_indices["masu"]:
        print("masu found.")
        sentence_ender = masu
        if  words[-3][-2:] == "ku": 
            print("adverb found")
            meaning_carrier = s.l @ particle_links @ s @ h.l
            adv_index = word_count - 3
        elif words[-3][-2:] == "te": # this could be improved "tte" is always the te-form.
            print("te-form found")   # this could be improved to catch adverbs before a te-form
            meaning_carrier = s.l @ s @ h.l
            te_index = word_count - 3
        elif words[-2] == "shi" and words[-3] != "wo": # handle special verb suru, which makes verbal nouns
            print("polite verbal noun found")
            meaning_carrier = n.r @ particle_links @ s @ h.l
            noun_verb_index = word_count - 3
        else:
            meaning_carrier = particle_links @ s @ h.l
    elif special_word_indices["da"]:
        print("da found.")
        sentence_ender = n.r @ particle_links @ s 
    elif words[-1][-1] == "i": 
        print("casual i-adjective sentence ender found.")
        sentence_ender = particle_links @ s
    elif words[-1][-1] == "u":
        print("the sentence ends in a casual verb.")
        if words[-2][-2:] == "ku":
            print("adverb found")
            sentence_ender = s.l @ particle_links @ s
            adv_index = word_count - 2
        elif words[-2][-2:] == "te":
            print("te-form found")
            sentence_ender = s.l @ s
            te_index = word_count - 2
        elif words[-1] == "suru" and words[-2] != "wo": # handle special verb suru, which makes verbal nouns
            print("casual verbal noun found")
            sentence_ender = n.r @ particle_links @ s
            noun_verb_index = word_count - 2
        else:
            sentence_ender = particle_links @ s
    else: # this will also catch dependent clauses and single word utterances. This could be improved.
        print("the text is a dependent clause, word, or phrase.") 
    
    # Third: Build out list of types that uses the same indices as the words
    types = [None] * word_count
    if sentence_ender:
        types[word_count - 1] = sentence_ender
        print(f"type word_count-1: {types[word_count - 1]}")
    if meaning_carrier:
        types[word_count - 2] = meaning_carrier
        print(f"type word_count-2: {types[word_count - 2]}")
    if special_word_indices["desu"] and sentence_ender == h @ n.r @ particle_links @ s:
        if types[word_count - 2] is None:
            types[word_count - 2] = pred_polite_n
        print(f"type word_count-2: {types[word_count - 2]}")
    if special_word_indices["da"]:
        if types[word_count - 2] is None:
            types[word_count - 2] = n
        print(f"type word_count-2: {types[word_count - 2]}")
    if te_index:
        types[te_index] = particle_links @ s.l.l
        print(f"te: {types[te_index]}")
    if adv_index:
        types[adv_index] = adv
        print(f"adv: {types[adv_index]}")
    if noun_verb_index:
        types[noun_verb_index] = n
        print(f"verbal_noun: {types[noun_verb_index]}")
    if special_word_indices["san"]:
        print(f"title: {special_word_indices['san']}")
        for title_index in special_word_indices["san"]:
            types[title_index] = title
            print(f"title type: {types[title_index]}")
            if title_index >= 1 and types[title_index - 1] is None:
                types[title_index - 1] = name
    if special_word_indices["no"]:
        print(f"possessive no: {special_word_indices['no']}")
        for no_index in special_word_indices["no"]:
            types[no_index] = possessive_no
            if no_index >= 1 and types[no_index - 1] is None:
                types[no_index - 1] = n
            if no_index < len(types) and types[no_index + 1] is None:
                types[no_index + 1] = n
    if particle_count: 
        flattened_particle_indices = [index for particle in particles if particle in special_word_indices for index in special_word_indices[particle]]
        flattened_particle_indices.sort() # particles must be assigned from left to right to avoid overwrite issues
        print(f"particle indices: {flattened_particle_indices}")
        for particle_index in flattened_particle_indices:
            if types[particle_index] == None: # some particles like "de" are also polite verb stems.
                types[particle_index] = particle_logical
                print(f"particle type: {types[particle_index]}")
            if particle_index >= 2 and types[particle_index - 2] is None:
                types[particle_index-2] = adj_i
            if particle_index >= 1 and types[particle_index - 1] is None:
                types[particle_index-1] = n
    print(f"types: {types}")
    sub_types = " @ ".join(str(type) for type in types).split(" @ ")
    print(f"sub_types: {sub_types}")
        
    # Fourth: assign types to words based on the sentence ender and the number of particles.
    word_type_dict = zip(words, types)
    typed_sentence = []
    for word, type in word_type_dict: 
        typed_sentence.append(Word(word, type))
    print(f"typed_sentence: {typed_sentence}")
    
    # Fifth: assign morphism connections programmatically based on the sub-type indices
    # try:
    sub_types_scratch = sub_types.copy()
    s_type = sub_types_scratch.index("s")
    morphisms = []
    sub_types_scratch[s_type] = True
    jump_distance = 1
    while len(set(sub_types_scratch)) > 1 and jump_distance < len(sub_types_scratch) - 1:
        for i in range (len(sub_types_scratch) - 1) :
            if sub_types_scratch[i] == "h.l" and sub_types_scratch[i + jump_distance] == "h":
                morphisms.append((Cup, i, i + jump_distance))
                sub_types_scratch[i] = True
                sub_types_scratch[i + jump_distance] = True
            elif sub_types_scratch[i] == "n.l" and sub_types_scratch[i + jump_distance] == "n":
                morphisms.append((Cup, i, i + jump_distance))
                sub_types_scratch[i] = True
                sub_types_scratch[i + jump_distance] = True
            elif sub_types_scratch[i] == "n" and sub_types_scratch[i + jump_distance] == "n.r":
                morphisms.append((Cup, i, i + jump_distance))
                sub_types_scratch[i] = True
                sub_types_scratch[i + jump_distance] = True
            elif sub_types_scratch[i] == "s.l.l" and sub_types_scratch[i + jump_distance] == "s.l":
                morphisms.append((Cup, i, i + jump_distance))
                sub_types_scratch[i] = True
                sub_types_scratch[i + jump_distance] = True
        print(f"Morphism loop: jump_distance {jump_distance} and morphisms {len(morphisms)}")
        jump_distance = jump_distance + 1
    # except: # diagram could not be constructed
    #     return None
    
    # Sixth: build and return diagram
    diagram = Diagram.create_pregroup_diagram(typed_sentence, morphisms)
    #diagram.draw(figsize=(9, 10))
    return diagram

## Quantizer Method
# This method takes a diagram and turns it into a normalized and simplified quantum circuit
def quantizer(diagram):
    diagram = diagram.normal_form()
    circuit = ansatz(remove_cups(diagram.normal_form()))
    #circuit.draw(figsize=(9,10))
    return circuit

## Procedure
# Test driver
# diagram = diagramizer("Suzuki san ga kirei desu")
# if diagram is not None:
#     circuit = quantizer(diagram)
    
# read in data and labels
train_labels, train_data = read_data('./datasets/shuffled/trainsetshuff.txt')
val_labels, val_data = read_data('./datasets/shuffled/valsetshuff.txt')
test_labels, test_data = read_data('./datasets/shuffled/testsetshuff.txt')

# use diagramizer to build diagrams
train_diagrams = [diagramizer(sentence) for sentence in train_data if sentence is not None]
val_diagrams = [diagramizer(sentence) for sentence in val_data if sentence is not None]
test_diagrams = [diagramizer(sentence) for sentence in test_data if sentence is not None]
print("Diagrams constructed from corpus data.")

# Rewrite diagrams so that they are the same shape and can be used with pytorch tensors.
train_diagrams = [u_cdom_rewriter(diagram) for diagram in train_diagrams if diagram is not None]
val_diagrams = [u_cdom_rewriter(diagram) for diagram in val_diagrams if diagram is not None]
test_diagrams = [u_cdom_rewriter(diagram) for diagram in test_diagrams if diagram is not None]
print("Diagrams padded.")

# create labeled maps of diagrams (this is not using the read in data directly)
train_labels = [label for (diagram, label) in zip(train_diagrams, train_labels) if diagram is not None]
val_labels = [label for (diagram, label) in zip(val_diagrams, val_labels) if diagram is not None]
test_labels = [label for (diagram, label) in zip(test_diagrams, test_labels) if diagram is not None]
print("Label lists constructed.")

# construct circuits
train_circuits = [quantizer(diagram) for diagram in train_diagrams if diagram is not None]
val_circuits = [quantizer(diagram) for diagram in val_diagrams if diagram is not None]
test_circuits = [quantizer(diagram) for diagram in test_diagrams if diagram is not None]
print("Circuits constructed from diagrams.")

# instantiate training model
all_circuits = train_circuits + val_circuits + test_circuits
backend_config = {'backend': 'default.qubit'}
model = PennyLaneModel.from_diagrams(all_circuits, probabilities=True, normalize=True, 
                                     backend_config=backend_config)
model.initialise_weights()
# TODO: make an if with a bool check to determine if running on real hardware
# qml.default_config['qiskit.ibmq.ibmqx_token'] = 'my_API_token'
# qml.default_config.save(qml.default_config.path)
# backend_config = {'ba'}

# create datasets
train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
val_dataset = Dataset(val_circuits, val_labels)

# initialize trainer
trainer = PytorchTrainer(model=model, loss_function=loss, optimizer=optimizer, learning_rate=LEARNING_RATE,
    epochs=EPOCHS, evaluate_functions={'acc': acc}, evaluate_on_train=True, use_tensorboard=False, 
    verbose='text', log_dir=log_dir, seed=SEED)

# train model
trainer.fit(train_dataset, val_dataset)

# visualize results
fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
ax_tl.set_title('Training set')
ax_tr.set_title('Validation set')
ax_bl.set_xlabel('Iterations')
ax_br.set_xlabel('Iterations')
ax_bl.set_ylabel('Accuracy')
ax_tl.set_ylabel('Loss')
colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
range_ = np.arange(1, len(trainer.train_epoch_costs)+1)
ax_tl.plot(range_, trainer.train_epoch_costs, color=next(colours))
ax_bl.plot(range_, trainer.train_eval_results['acc'], color=next(colours))
ax_tr.plot(range_, trainer.val_costs, color=next(colours))
ax_br.plot(range_, trainer.val_eval_results['acc'], color=next(colours))
# mark best model as circle
best_epoch = np.argmin(trainer.val_costs)
ax_tl.plot(best_epoch + 1, trainer.train_epoch_costs[best_epoch], 'o', color='black', fillstyle='none')
ax_tr.plot(best_epoch + 1, trainer.val_costs[best_epoch], 'o', color='black', fillstyle='none')
ax_bl.plot(best_epoch + 1, trainer.train_eval_results['acc'][best_epoch], 'o', color='black', fillstyle='none')
ax_br.plot(best_epoch + 1, trainer.val_eval_results['acc'][best_epoch], 'o', color='black', fillstyle='none')
ax_tr.text(best_epoch + 1.4, trainer.val_costs[best_epoch], 'early stopping', va='center')
plt.show()

# print test set accuracy
pred = model(test_circuits)
labels = torch.tensor(test_labels)
print('Final test accuracy: {}'.format(acc(pred, labels)))