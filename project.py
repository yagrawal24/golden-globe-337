import spacy
import re

# Load the SpaCy English language model
nlp = spacy.load('en_core_web_sm')

# Define regex patterns to capture various award-related keywords
win_keywords = r"(win|wins|won|winner|victor|victory|award recipient|achievement)"
present_keywords = r"(presented by|presenter|present)"
nominate_keywords = r"(nominated|nominee|nominations|in the running for)"
award_keywords = r"(best|award|category|achievement|lifetime)"
end_award_keywords = r"(winner|nominee|presented by|presenter)"  # Stop category extraction at these

def extract_entities(doc):
    """Extract named entities from the SpaCy doc."""
    return [(ent.text, ent.label_) for ent in doc.ents]

def is_name_match(person_name, subject_name):
    """Compare if parts of the names match (case insensitive)."""
    person_name = person_name.lower()
    subject_name = subject_name.lower()
    
    person_parts = person_name.split()
    subject_parts = subject_name.split()
    
    return any(part in person_parts for part in subject_parts)

def extract_subject(doc):
    """Extract the subject of the sentence using dependency parsing."""
    for token in doc:
        if token.dep_ == "nsubj":
            return token.text
    return None

def extract_award_category(doc):
    """Extract award category based on award keywords and stop when winner or similar is encountered."""
    award_category_tokens = []
    for token in doc:
        if re.search(award_keywords, token.text, re.IGNORECASE):
            # Add relevant tokens (e.g., nouns, adjectives)
            award_category_tokens.append(token.text)
        elif re.search(end_award_keywords, token.text, re.IGNORECASE):
            # Stop extraction if we encounter keywords like 'winner'
            break
    return ' '.join(award_category_tokens).strip() if award_category_tokens else None

def extract_award_details(text):
    """Identify winners, nominees, presenters, and award categories in the text."""
    doc = nlp(text)
    
    # Initialize holders for the capture groups
    winner = set()
    nominees = set()
    presenters = set()
    award_category = extract_award_category(doc)

    # Extract named entities and subjects
    entities = extract_entities(doc)
    subject = extract_subject(doc)

    # Determine the context based on subject and keywords
    if subject:
        # If the subject is a person, check their role based on nearby keywords
        for token in doc:
            if re.search(win_keywords, token.text, re.IGNORECASE):
                for ent in entities:
                    if ent[1] == "PERSON" and is_name_match(ent[0], subject):
                        winner.add(ent[0])
                        nominees.add(ent[0])  # If someone won, they must have been nominated
                        break
            
            elif re.search(present_keywords, token.text, re.IGNORECASE):
                for ent in entities:
                    if ent[1] == "PERSON" and is_name_match(ent[0], subject):
                        presenters.add(ent[0])
            
            elif re.search(nominate_keywords, token.text, re.IGNORECASE):
                for ent in entities:
                    if ent[1] == "PERSON" and is_name_match(ent[0], subject):
                        nominees.add(ent[0])

    # Return the captured groups
    return {
        "winner": list(winner) if winner else None,
        "nominees": list(nominees) if nominees else None,
        "presenters": list(presenters) if presenters else None,
        "award_category": award_category if award_category else None
    }

# Test the function
text1 = "rt @galaxiemag lifetime achievement award winner jodie foster is 50 she has been acting for 47 of those years amazing"
text2 = "jessica chastain best actress drama winner at golden globes 2013 presented by tom hanks"

result1 = extract_award_details(text1)
result2 = extract_award_details(text2)

print(result1)
print('\n')
print(result2)
