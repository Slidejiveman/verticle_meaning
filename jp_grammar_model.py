from lambeq import AtomicType
from lambeq.backend.grammar import Ty

## define the grammatical types
# note the "h" type for honorifics/polite form
n, s, h = AtomicType.NOUN, AtomicType.SENTENCE, Ty ('h') 

## define grammar model. 
# NOTE: There are extra types defined here to capture possible avenues of research
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