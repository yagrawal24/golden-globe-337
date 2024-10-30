import pandas as pd
import re
import spacy
import emoji
from ftfy import fix_text
import unidecode

# Step 1: Load the cleaned data (same cleaning process as before)
cleaned_data = pd.read_csv('text_cleaned.csv')['text']

# Step 2: Define nomination-related keywords and patterns
# These are words/phrases related to nominations, hoping to win, or being nominated
nominee_keywords = r'\bnominated\b|\bnominee\b|\bnomination\b|\bhope\b|\bwish\b|\bthink\b|\bbelieve\b|\bshould\b'

# Step 3: Filter the dataset to keep only tweets discussing nominations
nominee_data = cleaned_data[cleaned_data.apply(lambda x: re.search(nominee_keywords, x) != None)]

# Save the nominee-related tweets to a new CSV file
nominee_data.to_csv("nominees.csv")

# Step 4: Extract named entities (similar to the previous entity extraction for winners)
def extract_entities(doc):
    """Extract entities from the SpaCy doc."""
    return [(ent.text, ent.label_) for ent in doc.ents]

# Load the SpaCy model
nlp = spacy.load('en_core_web_sm')

def get_nominee(text):
    """Extract nominee-related entities from text."""
    doc = nlp(text)
    entities = extract_entities(doc)
    
    # Extract the subject (nsubj) or use PERSON entities as potential nominees
    subject = []
    for token in doc:
        if token.dep_ == 'nsubj':
            subject.append(token.text)
    
    # Fallback: If no subject found, use named entities like PERSON
    for ent, label in entities:
        if label in ['PERSON', 'ORG', 'WORK_OF_ART']:
            return ent
    return None

# Step 5: Apply the nominee extraction process and save it
spacy_size = 1500
spacy_data = nominee_data.head(spacy_size).to_frame()
spacy_data['nominee'] = spacy_data['text'].apply(get_nominee)

# Save the extracted nominee information to a CSV
spacy_data.to_csv('nominees_info.csv', index=False)
