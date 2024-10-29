import pandas as pd
import re
import spacy
import emoji
from ftfy import fix_text
import unidecode

# Load data
df = pd.read_json('gg2013.json')['text']
df.to_csv('text.csv')

def clean(text):
    text = fix_text(text)
    
    # Skip tweets that start with "rt @username"
    if re.match(r"^rt @\w+", text, re.IGNORECASE):
        return ""  # Mark as empty if it matches the "rt @username" pattern
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|pic.twitter\S+', '', text)
    
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    
    # Remove non-alphanumeric characters, except @ and spaces
    text = re.sub(r'[^A-Za-z0-9@ ]+', '', text)
    
    # Remove extra spaces
    text = re.sub(' +', ' ', text).strip()
    
    # Convert text to lowercase
    text = text.lower()
    
    return text

# Apply cleaning function
df = df.apply(clean)

# cleaned_data = df[df != ""]
cleaned_data = df[df.str.strip() != ""]
cleaned_data.to_csv('text_cleaned_2.csv')
cleaned_data[cleaned_data.apply(lambda x: re.search('(?=.*award|AWARD)(?=.*wins|Wins|WINS|winner|WINNER).*', x) != None)]

# filtered_data = cleaned_data[cleaned_data.apply(lambda x: bool(re.search(r'(?=.*award)(?=.*wins|winner)', x, re.IGNORECASE)))]
# filtered_data.to_csv('filtered_award_mentions.csv', index=False)

data = pd.read_csv('text_cleaned_2.csv')['text']

win_data = cleaned_data[cleaned_data.apply(lambda x: re.search(r'\bwins\b', x) != None)]

win_data.to_csv("wins_no_at.csv")

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