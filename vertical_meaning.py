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

## Define special words
particles = ["ha", "ga", "ni", "de", "wo"]
titles = ["san"]
honorific_words = ["desu", "masu"] + titles
casual_copula = ["da"]
possessives = ["no"]
special_words = casual_copula + honorific_words + particles + possessives

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
            print("verbal noun found")
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
        else:
            sentence_ender = particle_links @ s
    else: # this will also catch dependent clauses and single word utterances. This could be improved.
        print("the text is a dependent clause, word, or phrase.") 
    
    # Third: Build out list of types that uses the same indices as the words
    types = [None] * word_count
    if sentence_ender: # TODO: doesn't account for desu
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
        print(f"particle indices: {flattened_particle_indices}")
        for particle_index in flattened_particle_indices:
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

# Use diagramizer to build diagrams
train_diagrams = []
for sentence in train_data:
    train_diagrams.append(diagramizer(sentence))
print(train_diagrams[:5])