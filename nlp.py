# import spacy
# from spacy import displacy
# import cairosvg
# import re

# # load english language model
# nlp = spacy.load('en_core_web_sm')
# # text = "This is a sample sentence."

# # create spacy
# # doc = nlp(text)

# '''
# Determine Parts of Speech
# '''
# # for token in doc:
# #     print(token.text, '->', token.pos_)

# '''
# RESULT:

# This -> PRON
# is -> AUX
# a -> DET
# sample -> NOUN
# sentence -> NOUN
# . -> PUNCT
# '''

# '''
# Extract Nouns
# '''
# # for token in doc:
# #     # check token pos
# #     if token.pos_ == 'NOUN':
# #         # print token
# #         print(token.text)

# '''
# RESULT:

# sample
# sentence
# '''

# # text = "The children love cream biscuits"

# # create spacy
# # doc = nlp(text)

# # for token in doc:
# #     print(token.text, '->', token.pos_)

# '''
# RESULT:

# The -> DET
# children -> NOUN
# love -> VERB
# cream -> NOUN
# biscuits -> NOUN
# '''

# # for token in doc:
# #     # extract subject
# #     if (token.dep_ == 'nsubj'):
# #         print(token.text)
# #     # extract object
# #     elif (token.dep_ == 'dobj'):
# #         print(token.text)

# '''
# RESULT:

# children
# biscuits
# '''

# '''
# Dependency Graph

# 1. The arrowhead points to the words that are dependent on the word pointed by the origin of the arrow
# 2. The former is referred to as the child node of the latter. For example, “children” is the child node of “love”
# 3. The word which has no incoming arrow is called the root node of the sentence
# '''
# # displacy.render(doc, style = 'dep', jupyter = True)

# # Render the dependency tree as SVG
# # svg = displacy.render(doc, style='dep', jupyter=False, options={'compact': True})

# # Save the SVG content to a file
# # with open("dependency_parse.svg", "w", encoding="utf-8") as file:
# #     file.write(svg)

# # print("Dependency parse saved as 'dependency_parse.svg'")

# # Convert the SVG to PNG
# # cairosvg.svg2png(url="dependency_parse.svg", write_to="dependency_parse.png")

# # print("Dependency parse saved as 'dependency_parse.png'")

# '''
# PROJECT 1 TEST
# '''

# text = "rt @galaxiemag lifetime achievement award winner jodie foster is 50 she has been acting for 47 of those years amazing"

# # create spacy
# doc = nlp(text)

# # for token in doc:
# #     print(token.text, '->', token.pos_)
# '''
# rt -> INTJ
# @galaxiemag -> PROPN
# lifetime -> NOUN
# achievement -> NOUN
# award -> NOUN
# winner -> NOUN
# jodie -> PROPN
# foster -> PROPN
# is -> AUX
# 50 -> NUM
# she -> PRON
# has -> AUX
# been -> AUX
# acting -> VERB
# for -> ADP
# 47 -> NUM
# of -> ADP
# those -> DET
# years -> NOUN
# amazing -> ADJ
# '''

# # displacy.render(doc, style = 'dep', jupyter = True)

# # svg = displacy.render(doc, style='dep', jupyter=False, options={'compact': True})

# # with open("test.svg", "w", encoding="utf-8") as file:
# #     file.write(svg)

# # print("Dependency parse saved as 'test.svg'")

# # cairosvg.svg2png(url="test.svg", write_to="test.png")

# # print("Dependency parse saved as 'test.png'")

# # for token in doc:
# #     # extract subject
# #     if (token.dep_ == 'nsubj'):
# #         print(token.text)
# #     # extract object
# #     elif (token.dep_ == 'dobj'):
# #         print(token.text)
# '''
# foster
# she
# '''

# # print(re.findall(r"(.+) (wins|Wins|won|Won|winner|Winner|winner|Winners) (.+)", text))
# '''
# [('rt @galaxiemag lifetime achievement award', 'winner', 'jodie foster is 50 she has been acting for 47 of those years amazing')]
# '''

# def extract_entities(text):
#     doc = nlp(text)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities

# # print(extract_entities(text))

# '''
# [('jodie foster', 'PERSON'), ('50', 'DATE'), ('47', 'CARDINAL')]
# '''

# # for i in extract_entities(text):
# #     print(i)
# '''
# ('jodie foster', 'PERSON')
# ('50', 'DATE')
# ('47', 'CARDINAL')
# '''

# # for i in extract_entities(text):
# #     if i[1] == "PERSON":
# #         print(i)
# '''
# ('jodie foster', 'PERSON')
# '''

# # for token in doc:
# #     if (token.dep_ == 'nsubj'):
# #         nsubj = token.text
# #     elif (token.dep_ == 'dobj'):
# #         dobj = token.text

# alleged_winner = ""
# subject = []
# actual_winner = ""

# for ent in extract_entities(text):
#     if ent[1] == "PERSON":
#         alleged_winner = ent[0]

# for token in doc:
#     # extract subject
#     if (token.dep_ == 'nsubj'):
#         subject.append(token.text)

# # print(subject)
# # print(alleged_winner)

import spacy

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

# def extract_full_name(sentence):
#     """Extract full names from a given sentence."""
#     # Process the sentence using the SpaCy model
#     doc = nlp(sentence)
    
#     # Extract entities of type 'PERSON'
#     full_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
#     return full_names

# # Example usage
# sentence = "Chris Evans won the award, while Scarlett Johansson was nominated."
# full_names = extract_full_name(sentence)

# print(full_names)  # Output: ['Chris Evans', 'Scarlett Johansson']

# text = 'rt @kriskling pretty sweet they had president clinton introduce the movie lincoln nominated for best motion picture drama'
'''
rt: npadvmod
@kriskling: acl
pretty: advmod
sweet: acomp
they: nsubj
had: ROOT
president: compound
clinton: nsubj
introduce: ccomp
the: det
movie: compound
lincoln: dobj
nominated: acl
for: prep
best: amod
motion: compound
picture: compound
drama: pobj
'''

text = 'rt @perezhilton @benaffleck argo wins best drama at the golden globes'
'''
rt: nmod
@perezhilton: compound
@benaffleck: compound
argo: nsubj
wins: ROOT
best: amod
drama: dobj
at: prep
the: det
golden: amod
globes: pobj
'''

def extract_nominee(text):
    doc = nlp(text)

    for token in doc:
        # if (token.dep_ == 'nsubj'):
            # print(f'{token}: {token.dep_}')
        print(f'{token}: {token.dep_}')

extract_nominee(text)