import spacy
import re
import pandas as pd

# Load the SpaCy English language model
nlp = spacy.load('en_core_web_sm')

# Define regex pattern to capture award-related keywords
win_keywords = r"(\bwin\b|\bwins\b|\bwon\b|\bawarded\b)"

def extract_entities(doc):
    """Extract named entities from the SpaCy doc."""
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_subject_as_nominee(doc):
    """Extract the subject (nsubj) of the sentence, which is often the winner/nominee."""
    for token in doc:
        if token.dep_ == 'nsubj' and token.head.text in ['wins', 'won', 'receives']:
            return token.text
    return None

def extract_award_name_after_best(doc):
    """Extract the full award name starting from 'best' using dependency parsing."""
    for token in doc:
        if token.text.lower() == 'best':
            award_tokens = [token]
            for right_token in doc[token.i + 1:]:
                if right_token.pos_ in ('PUNCT', 'CCONJ', 'VERB', 'ADP', 'DET') or right_token.dep_ == 'punct':
                    break
                award_tokens.append(right_token)
            award_name = ' '.join([t.text for t in award_tokens])
            return award_name
    return None

def clean_text(text):
    """Remove 'rt' and any user mentions that start with '@' from the text."""
    # Remove "rt" at the start
    text = re.sub(r'^rt\s+', '', text, flags=re.IGNORECASE)
    # Remove user mentions (strings starting with '@')
    text = re.sub(r'@\w+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_award_winner(text):
    """Attempt to extract the person, song, or movie that won an award based on text."""
    
    # Clean the text by removing 'rt' and user mentions
    cleaned_text = clean_text(text)
    
    doc = nlp(cleaned_text)
    
    # Check if the tweet mentions winning or awards
    if re.search(win_keywords, cleaned_text, re.IGNORECASE):
        # Try to extract the winner or nominee from subject (nsubj)
        nominee = extract_subject_as_nominee(doc)
        award = extract_award_name_after_best(doc)

        if nominee and award:
            return f"{nominee} | {award} | winner"
        elif award:
            return f"None | {award} | winner"
    
    return None

# Test the function
text = 'rt @perezhilton @benaffleck argo wins best drama at the golden globes'
print(find_award_winner(text))

# Load the dataset and apply the function
win_data = pd.read_csv('wins.csv')['text']
output = win_data.apply(lambda x: find_award_winner(x))

# Save the output
output_df = pd.DataFrame({'text': win_data, 'extracted_award': output})
output_df.to_csv('award_names.csv', index=False)
