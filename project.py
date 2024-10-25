# import pandas as pd
# import re
# import spacy

# # Load SpaCy model
# nlp = spacy.load('en_core_web_sm')

# # Load data
# win_data = pd.read_csv('wins.csv')['text']
# spacy_size = 200  # Processing top 200 entries for SpaCy
# test = win_data.head(spacy_size).to_frame()

# # Define regex pattern to capture award-related keywords
# win_keywords = r"(\bwin\b|\bwins\b|\bwon\b|\bwinner\b|\bawarded\b|\breceive\b|\breceived\b)"

# def extract_entities(doc):
#     """Extract named entities from the SpaCy doc, excluding entities starting with '@' or 'rt @'."""
#     return [
#         (ent.text, ent.label_) for ent in doc.ents
#         if ent.label_ == "PERSON" and not (ent.text.startswith('@') or ent.text.lower().startswith('rt @'))
#     ]

# def is_name_match(person_name, subject_name):
#     """Check if parts of the names match (case insensitive)."""
#     person_parts = person_name.lower().split()
#     subject_parts = subject_name.lower().split()
#     return any(part in person_parts for part in subject_parts)

# def extract_award_category(doc):
#     """Extract award category by finding descriptive tokens around the winning keyword."""
#     award_tokens = []
#     capturing = False
#     for token in doc:
#         # Only add to award tokens if token doesn't start with "@" or "rt @"
#         if token.dep_ in {'amod', 'compound', 'dobj'} and token.pos_ in {'ADJ', 'NOUN'}:
#             if not token.text.startswith('@') and not token.text.lower().startswith('rt @'):
#                 award_tokens.append(token.text)
#                 capturing = True
#         elif capturing and token.dep_ not in {'amod', 'compound', 'dobj'}:
#             break
#     return " ".join(award_tokens) if award_tokens else None

# def is_winner_keyword_near_person(doc, person_name):
#     """Check if win-related keywords appear near a PERSON entity in the text."""
#     for token in doc:
#         if re.search(win_keywords, token.text, re.IGNORECASE):
#             for ent in doc.ents:
#                 if ent.label_ == "PERSON" and person_name.lower() in ent.text.lower():
#                     if abs(token.i - ent.start) <= 5:  # Check within 5 tokens for proximity
#                         return True
#     return False

# def find_award_winner(text):
#     """Extract award information from text."""
#     doc = nlp(text)
    
#     # Check if the tweet mentions winning or awards
#     if re.search(win_keywords, text, re.IGNORECASE):
        
#         winners = []
#         entities = extract_entities(doc)
#         award_category = extract_award_category(doc)
        
#         # Identify all PERSON entities near win-related keywords
#         for ent in entities:
#             if ent[1] == "PERSON" and is_winner_keyword_near_person(doc, ent[0]):
#                 winners.append(ent[0])
        
#         # Return structured dictionary with possible award information
#         if winners:
#             return {
#                 "winner": winners,
#                 "nominees": winners,
#                 "presenters": [],
#                 "award_category": award_category or "N/A"
#             }
    
#     return {"winner": None, "nominees": None, "presenters": None, "award_category": None}

# # Apply function and save to CSV
# test['spacy'] = test['text'].apply(find_award_winner)
# test.to_csv('results.csv', index=False)

''''''
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
    """Extract contiguous PERSON entities, including multi-token names."""
    entities = []
    temp_entity = []

    for ent in doc.ents:
        if ent.label_ == "PERSON" and not (ent.text.startswith('@') or ent.text.lower().startswith('rt @')):
            if temp_entity and ent.start == temp_entity[-1].end:
                temp_entity.append(ent)  # Add contiguous name parts
            else:
                if temp_entity:
                    entities.append((" ".join([e.text for e in temp_entity]), "PERSON"))
                temp_entity = [ent]  # Start new entity

    # Append last entity if exists
    if temp_entity:
        entities.append((" ".join([e.text for e in temp_entity]), "PERSON"))

    return entities

def extract_award_category(text, doc):
    """Extract award category by first using regex to find a match, and then using SpaCy token parsing."""
    # Step 1: Try to extract award category using regex for common patterns
    # regex_pattern = re.compile(r"(?:(?:\w+\s)*?)(?:win|wins|won|winner|awarded|receive|received|gets)(?:\s+(.*?))(?:\s*http)?")
    # match = regex_pattern.search(text)
    # if match:
    #     return match.group(1)  # Return if regex finds a match

    # Step 2: Fallback - Use SpaCy dependency parsing for additional flexibility
    award_tokens = []
    capturing = False
    for token in doc:
        # Only add to award tokens if token doesn't start with "@" or "rt @"
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
    
    # Check if the tweet mentions winning or awards
    if re.search(win_keywords, text, re.IGNORECASE):
        winners = []
        entities = extract_entities(doc)
        award_category = extract_award_category(text, doc)  # Pass both text and doc here

        # Identify all PERSON entities
        for ent in entities:
            if re.search(win_keywords, text, re.IGNORECASE):
                winners.append(ent[0])  # Append full name
        
        # Return structured dictionary with possible award information
        if winners:
            return {
                "winner": winners,
                "nominees": winners,  # This could be further refined if more data is available
                "presenters": [],
                "award_category": award_category or "N/A"
            }

    return {"winner": None, "nominees": None, "presenters": None, "award_category": None}

# Apply function and save to CSV
test['spacy'] = test['text'].apply(find_award_winner)
test.to_csv('results.csv', index=False)
