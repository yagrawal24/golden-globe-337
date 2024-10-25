import pandas as pd
import re
import spacy

cleaned_data = pd.read_csv('text_cleaned.csv')['text']

# win_data = cleaned_data[cleaned_data.apply(lambda x: re.search(r'\b(win|wins|won|winner|victor|victory|receive|received|receives|awarded)\b', x) != None)]
win_data = cleaned_data[cleaned_data.apply(lambda x: re.search(r'\bwins\b', x) != None)]

win_data.to_csv("wins.csv")

def extract_entities(doc):
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def pattern(word, text): 
    pat = re.compile(re.escape(word), re.IGNORECASE) 
    return bool(re.search(pat, text))

nlp = spacy.load('en_core_web_sm')
def get_winner(text):
    alleged_winner = ""
    subject = []
    actual_winner = ""
    
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # return entities
    
    # for ent in entities:
    #     if ent[1] == "PERSON":
    #         alleged_winner = ent[0]
            
    for token in doc:
        # extract subject
        if (token.dep_ == 'nsubj'):
            subject.append(token.text)
    
    entities = extract_entities(doc)
    for s in subject:
        for e, _ in entities:
            if pattern(s, e):
                return e

spacy_size = 1500
spacy_data = win_data.head(spacy_size).to_frame()
spacy_data['spacy'] = win_data.head(spacy_size).apply(get_winner)
spacy_data.to_csv('spacy_info.csv')