import pandas as pd
import re
import spacy

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load data
win_data = pd.read_csv('wins.csv')['text']
cleaned_data = pd.read_csv('text_cleaned.csv')['text']
spacy_data = pd.read_csv('spacy_info.csv')
spacy_size = 200  # Processing top 200 entries for SpaCy
test = win_data.head(spacy_size).to_frame()

# Define regex pattern to capture award-related keywords
win_keywords = r"(win|wins|won|winner|awarded|receive|received|gets)"

# def extract_entities(doc):
#     """Extract named entities from the SpaCy doc, excluding entities starting with '@' or 'rt @'."""
#     return [
#         (ent.text, ent.label_) for ent in doc.ents
#         if ent.label_ == "PERSON" and not (ent.text.startswith('@') or ent.text.lower().startswith('rt @'))
#     ]

def extract_entities(doc):
    """Extract named entities from the SpaCy doc, ensuring full names are captured."""
    return [
        ent.text for ent in doc.ents
        if ent.label_ == "PERSON" and not (ent.text.startswith('@') or ent.text.lower().startswith('rt @'))
    ]

def is_name_match(person_name, subject_name):
    """Compare if parts of the names match (case insensitive)."""
    person_parts = person_name.lower().split()
    subject_parts = subject_name.lower().split()
    return any(part in person_parts for part in subject_parts)

def is_winner_keyword_near_person(doc, person_name):
    """Check if win-related keywords appear near a PERSON entity in the text."""
    for token in doc:
        if re.search(win_keywords, token.text, re.IGNORECASE):
            for ent in doc.ents:
                if ent.label_ == "PERSON" and person_name.lower() in ent.text.lower():
                    if abs(token.i - ent.start) <= 3 or abs(token.i - ent.end) <= 3:
                        return True
    return False

def extract_award_category(text, doc):
    """Extract award category using regex and SpaCy noun phrase parsing."""
    win_match = re.search(win_keywords, text, re.IGNORECASE)
    if win_match:
        win_index = win_match.end()
        words_after = text[win_index:].split()[:4]
        
        if "best" in [word.lower() for word in words_after]:
            award_candidates = [chunk.text for chunk in doc.noun_chunks if "best" in chunk.text.lower()]
            return award_candidates[0] if award_candidates else None

    award_tokens = []
    capturing = False
    for token in doc:
        if token.dep_ in {'amod', 'compound', 'dobj'} and token.pos_ in {'ADJ', 'NOUN'}:
            if not token.text.startswith('@') and not token.text.lower().startswith('rt @'):
                award_tokens.append(token.text)
                capturing = True
        elif capturing and token.dep_ not in {'amod', 'compound', 'dobj'}:
            break
    return " ".join(award_tokens) if award_tokens else None

def find_award_winner(text):
    """Extract award information from text."""
    doc = nlp(text)
    if re.search(win_keywords, text, re.IGNORECASE):
        winners = []
        entities = extract_entities(doc)
        award_category = extract_award_category(text, doc)

        for ent in entities:
            if is_winner_keyword_near_person(doc, ent):
                winners.append(ent)
        
        return {
            "winner": winners or None,
            "nominees": winners or None,
            "presenters": [],
            "award_category": award_category or "N/A"
        }

    return {"winner": None, "nominees": None, "presenters": None, "award_category": None}

# Example usage with test DataFrame
test['spacy'] = test['text'].apply(find_award_winner)
test.to_csv('results.csv', index=False)