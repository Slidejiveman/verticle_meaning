from lambeq.backend.grammar import Cup, Diagram, Word
from lambeq import RemoveCupsRewriter
import logging
import numpy as np
import matplotlib.pyplot as plt
from jp_grammar_model import * # grammar types

## utilities
remove_cups = RemoveCupsRewriter()
logging.basicConfig(filename='./logs/utilities/diagramizer.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
 

## helper functions
# this function finds indices of particles, sentence enders, and titles.
# these indices are used to type the other words in the sentence based on grammatical links/rules
def is_valid_input(sentence):
    endings = ("desu", "masu", "da")
    is_valid = False
    # this check could be improved, but it is sufficient for the dataset in question 
    if sentence is not None and sentence != "" and (sentence.endswith(endings) or sentence[-1] == "u" or (
                                                                sentence[-1] == "i" and sentence[-1] != "ei")):
        is_valid = True
    return is_valid

def find_special_words(words):
    special_word_indices = {special_word: [] for special_word in special_words}
    for index, special_word in enumerate(words):
        if special_word in special_word_indices:
            special_word_indices[special_word].append(index)
    logging.info("Indices and counts of special words: ")
    for special_word, special_word_index_list in special_word_indices.items():
        logging.info(f"{special_word}: {special_word_index_list}, Count: {len(special_word_index_list)}")
    return special_word_indices

# this function determins the number of particles in the sentence so that the verb type can be constructed later
def find_particle_count(special_word_indices):
    particle_count = sum(len(special_word_indices[particle]) for particle in particles if particle in special_word_indices)
    logging.info(f"particle count: {particle_count}")
    return particle_count

# build particle_links type, which will be used as a part of some sentence enders.
def build_particle_links(particle_count):
    particle_links = Ty()
    for _ in range(0, particle_count):
        particle_links = particle_links @ s.l
    logging.info(f"particle_links value: {particle_links}")
    return particle_links

# type the predicate of the sentence. Relies on helpers
def type_predicate(special_word_indices, words, particle_links):
    sentence_ender = None
    meaning_carrier = None
    adv_index = None
    te_index = None
    noun_verb_index = None
    if special_word_indices["desu"]:
        sentence_ender, meaning_carrier = type_desu_predicate(words, particle_links)
    elif special_word_indices["masu"]:
        sentence_ender, meaning_carrier, adv_index, te_index, noun_verb_index = type_masu_predicate(
                                                                                words, particle_links)
    elif special_word_indices["da"]:
        logging.info("da found.")
        sentence_ender = n.r @ particle_links @ s 
    elif words[-1][-1] == "i" and words[-1][-2:] != "ei": 
        logging.info("casual i-adjective sentence ender found.")
        sentence_ender = particle_links @ s
    elif words[-1][-1] == "u":
        sentence_ender, adv_index, te_index, noun_verb_index = type_casual_verb_predicate(
                                                                        words, particle_links)
    else: # this will also catch dependent clauses and single word utterances. This could be improved.
        logging.warning("Standard predicate types not found in type_predicate.")
    return sentence_ender, meaning_carrier, adv_index, te_index, noun_verb_index

# handle when the predicate contains "desu"
def type_desu_predicate(words, particle_links):
    logging.info("desu found.")
    # it is possible that the following check will catch a name like "Genji".
    # for now, this assumption is safe enough to work with. Filtering for common names could improve this.
    if words[-2][-1] == "i" and words[-2][-2:] != "ei":
        sentence_ender = desu_i_adj
        meaning_carrier = particle_links @ s @ h.l
    else:
        sentence_ender = h @ n.r @ particle_links @ s
        meaning_carrier = None
    return sentence_ender, meaning_carrier

# handle when the predicate contains "masu"
def type_masu_predicate(words, particle_links):
    logging.info("masu found.")
    word_count = len(words)
    sentence_ender = masu
    meaning_carrier = None
    adv_index = None
    te_index = None
    noun_verb_index = None
    if  words[-3][-2:] == "ku": 
        logging.info("adverb found")
        meaning_carrier = s.l @ particle_links @ s @ h.l
        adv_index = word_count - 3
    elif words[-3][-2:] == "te": # this could be improved "tte" is always the te-form.
        logging.info("te-form found")   # this could be improved to catch adverbs before a te-form
        meaning_carrier = s.l @ s @ h.l
        te_index = word_count - 3
    elif words[-2] == "shi" and words[-3] != "wo": # handle special verb suru, which makes verbal nouns
        logging.info("polite verbal noun found")
        meaning_carrier = n.r @ particle_links @ s @ h.l
        noun_verb_index = word_count - 3
    else:
        meaning_carrier = particle_links @ s @ h.l
    return sentence_ender, meaning_carrier, adv_index, te_index, noun_verb_index

# handlle when the predicate contains a casual verb
def type_casual_verb_predicate(words, particle_links):
    logging.info("the sentence ends in a casual verb.")
    word_count = len(words)
    adv_index = None
    te_index = None
    noun_verb_index = None
    if words[-2][-2:] == "ku":
        logging.info("adverb found")
        sentence_ender = s.l @ particle_links @ s
        adv_index = word_count - 2
    elif words[-2][-2:] == "te":
        logging.info("te-form found")
        sentence_ender = s.l @ s
        te_index = word_count - 2
    elif words[-1] == "suru" and words[-2] != "wo": # handle special verb suru, which makes verbal nouns
        logging.info("casual verbal noun found")
        sentence_ender = n.r @ particle_links @ s
        noun_verb_index = word_count - 2
    else:
        sentence_ender = particle_links @ s
    return sentence_ender, adv_index, te_index, noun_verb_index

# types all indexed words
def type_indexed_words(words, special_word_indices, sentence_ender, meaning_carrier, 
                       particle_links, particle_count, te_index, adv_index, noun_verb_index): 
    word_count = len(words)
    types = [None] * word_count
    if sentence_ender:
        types[word_count - 1] = sentence_ender
        logging.info(f"type word_count-1: {types[word_count - 1]}")
    if meaning_carrier:
        types[word_count - 2] = meaning_carrier
        logging.info(f"type word_count-2: {types[word_count - 2]}")
    if special_word_indices["desu"] and sentence_ender == h @ n.r @ particle_links @ s:
        if types[word_count - 2] is None:
            types[word_count - 2] = pred_polite_n
        logging.info(f"type word_count-2: {types[word_count - 2]}")
    if special_word_indices["da"]:
        if types[word_count - 2] is None:
            types[word_count - 2] = n
        logging.info(f"type word_count-2: {types[word_count - 2]}")
    if te_index:
        types[te_index] = particle_links @ s.l.l
        logging.info(f"te: {types[te_index]}")
    if adv_index:
        types[adv_index] = adv
        logging.info(f"adv: {types[adv_index]}")
    if noun_verb_index:
        types[noun_verb_index] = n
        logging.info(f"verbal_noun: {types[noun_verb_index]}")
    if special_word_indices["san"]:
        logging.info(f"title: {special_word_indices['san']}")
        for title_index in special_word_indices["san"]:
            types[title_index] = title
            logging.info(f"title type: {types[title_index]}")
            if title_index >= 1 and types[title_index - 1] is None:
                types[title_index - 1] = name
    if special_word_indices["no"]:
        logging.info(f"possessive no: {special_word_indices['no']}")
        for no_index in special_word_indices["no"]:
            types[no_index] = possessive_no
            if no_index >= 1 and types[no_index - 1] is None:
                types[no_index - 1] = n
            if no_index < len(types) and types[no_index + 1] is None:
                types[no_index + 1] = n
    if particle_count: 
        flattened_particle_indices = [index for particle in particles if particle in special_word_indices for index in special_word_indices[particle]]
        flattened_particle_indices.sort() # particles must be assigned from left to right to avoid overwrite issues
        logging.info(f"particle indices: {flattened_particle_indices}")
        for particle_index in flattened_particle_indices:
            if types[particle_index] == None: # some particles like "de" are also polite verb stems.
                types[particle_index] = particle_logical
                logging.info(f"particle type: {types[particle_index]}")
            if particle_index >= 2 and types[particle_index - 2] is None:
                types[particle_index-2] = adj_i
            if particle_index >= 1 and types[particle_index - 1] is None:
                types[particle_index-1] = n
    return types

# type the sentence based on its words and their associated types
def type_sentence(words, types):
    word_type_dict = zip(words, types)
    typed_sentence = []
    for word, type in word_type_dict: 
        typed_sentence.append(Word(word, type))
    logging.info(f"typed_sentence: {typed_sentence}")
    return typed_sentence

# create scratch sub-type list for the morphism algorithm.
# "True" means that the sub-type has been paired or should not be paired.
def create_type_scratch(sub_types):
    sub_types_scratch = sub_types.copy()
    try: 
        s_type = sub_types_scratch.index("s")
    except:
        raise ValueError("No 's' type was found in the sentence. Cannot create diagram.")
    sub_types_scratch[s_type] = True
    return sub_types_scratch

# check jump_distance to ensure while loop is not infinite
# in the case that not all of the types are matched
def is_safe_jump(sub_types_count, jump_distance):
    return jump_distance < sub_types_count - 1

# check if list index is safe
def is_safe_index(sub_types_count, i, jump_distance):
    is_safe = False
    if i + jump_distance < sub_types_count - 1:
        is_safe = True
    return is_safe

# link two connectable sub-types with a cup
def link_sub_types_with_cup(morphisms, sub_types_scratch, i, jump_distance):
    morphisms.append((Cup, i, i + jump_distance))
    sub_types_scratch[i] = True
    sub_types_scratch[i + jump_distance] = True
    
# constructs morphisms based on the sub-types of the words in the sentence
def build_morphisms(sub_types_scratch):
    sub_types_count = len(sub_types_scratch)
    morphisms = []
    jump_distance = 1
    while len(set(sub_types_scratch)) > 1 and is_safe_jump(sub_types_count, jump_distance):
        for i in range (len(sub_types_scratch) - 1) :
            if sub_types_scratch[i] == "h.l" and sub_types_scratch[i + jump_distance] == "h":
                link_sub_types_with_cup(morphisms, sub_types_scratch, i, jump_distance)
            elif sub_types_scratch[i] == "n.l" and sub_types_scratch[i + jump_distance] == "n":
                link_sub_types_with_cup(morphisms, sub_types_scratch, i, jump_distance)
            elif sub_types_scratch[i] == "n" and sub_types_scratch[i + jump_distance] == "n.r":
                link_sub_types_with_cup(morphisms, sub_types_scratch, i, jump_distance)
            elif sub_types_scratch[i] == "s.l.l" and sub_types_scratch[i + jump_distance] == "s.l":
                link_sub_types_with_cup(morphisms, sub_types_scratch, i, jump_distance)
        logging.info(f"Morphism loop: jump_distance {jump_distance} and morphisms {len(morphisms)}")
        jump_distance = jump_distance + 1
    return morphisms

## Diagramizer Algorithm
# Receives a sentence with tokens separated by spaces.
# Example supported sentences: "kanojo ga kaeru", "kanojo ga kaeri masu", "kanojo ga kami wo ori masu", 
# "kanojo ga tsukue de kami wo ori masu", "kanojo ga tsukue de kami wo oru", "kanojo ga utsukushii"
# "hon ga utsukushii desu", "Suzuki san ga utsukushii", "Suzuki san ga Tanaka san wo hanasu"
# "kanojo ga kami wo utsukushiku oru", "kanojo ga booru wo motte ki masu", "kanojo ga booru wo motte kuru",
# "hon ga kantan da", "Suzuki san ga kirei desu", and ""
def diagramizer(sentence):    
    try:
        diagram = None
        if is_valid_input(sentence): 
            # Put the input sentence into an ordered list of words
            words = sentence.split() 
            logging.info(words)
        else:
            raise Exception("Input must be a complete sentence separated by spaces with no punctuation marks.")
        
        # Determine positions of special words to aid in detecting the presence of adjectives or adverbs
        special_word_indices = find_special_words(words)
        
        ## Assign types to substrings to turn them into a list of Word types
        # First: Determine the number of particles present and build particle links type
        particle_count = find_particle_count(special_word_indices)
        particle_links = build_particle_links(particle_count)
        
        # Second: Check ending word for "desu", "masu", "da", or a word that ends in "i".
        sentence_ender, meaning_carrier, adv_index, te_index, noun_verb_index = type_predicate(
                                                                    special_word_indices, words, particle_links)
        
        # Third: assign types to words based on the sentence ender, the number of particles,
        # and special word indices. Finally, type the sentence
        types = type_indexed_words(words, special_word_indices, sentence_ender, meaning_carrier, 
                        particle_links, particle_count, te_index, adv_index, noun_verb_index)
        logging.info(f"types: {types}")
        typed_sentence = type_sentence(words, types)
        
        # Fourth: create sub-type indices so that morphisms can be drawn
        sub_types = " @ ".join(str(type) for type in types).split(" @ ")
        logging.info(f"sub_types: {sub_types}")
        
        # Fifth: assign morphism connections programmatically based on the sub-type indices
        sub_types_scratch = create_type_scratch(sub_types)
        morphisms = build_morphisms(sub_types_scratch)
        # Sixth: build and return diagram
        diagram = Diagram.create_pregroup_diagram(typed_sentence, morphisms)
    except ValueError as e:
        print(f"Caught an exception: {e}")
        logging.ERROR(f"Caught an exception: {e}")
    except Exception as e: 
        print(f"Caught an exception: {e}")
        logging.ERROR(f"Caught an exception: {e}")
    finally:
        return diagram

## Quantizer Method
# This method takes a diagram and turns it into a normalized and simplified quantum circuit
def quantizer(diagram, ansatz):
    diagram = diagram.normal_form()
    circuit = ansatz(remove_cups(diagram.normal_form()))
    return circuit 

## Default experiment visualization method
def visualizer(trainer):
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