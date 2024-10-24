import spacy
import re

# Load the SpaCy English language model
nlp = spacy.load('en_core_web_sm')

# Example tweet text
text = "rt @galaxiemag lifetime achievement award winner jodie foster is 50 she has been acting for 47 of those years amazing"
# text = "jessica chastain best actress drama winner at golden globes 2013"

# Use SpaCy's pipeline on the text
doc = nlp(text)

# Define a regex pattern to capture award-related words
# win_keywords = r"(wins|winner|awarded|received)"
win_keywords = r"(win|wins|won|winner|victor|victory|receive|received|receives|awarded)"

def extract_entities(doc):
    """Extract named entities from the SpaCy doc."""
    return [(ent.text, ent.label_) for ent in doc.ents]

def is_name_match(person_name, subject_name):
    """Compare if parts of the names match (case insensitive)."""
    # Normalize both names by converting to lowercase
    person_name = person_name.lower()
    subject_name = subject_name.lower()
    
    # Split names into components (first, last name) and check if there's overlap
    person_parts = person_name.split()
    subject_parts = subject_name.split()
    
    # Return True if any part of the subject matches any part of the person name
    return any(part in person_parts for part in subject_parts)

def is_winner_keyword_near_person(doc, person_name):
    """Check if win-related keywords appear near a PERSON entity in the text."""
    for token in doc:
        if re.search(win_keywords, token.text, re.IGNORECASE):
            # Check proximity (if the win-related word is close to the PERSON entity)
            for ent in doc.ents:
                if ent.label_ == "PERSON" and person_name.lower() in ent.text.lower():
                    # Check if the PERSON and win keyword are within 3 words of each other
                    if abs(token.i - ent.start) <= 3 or abs(token.i - ent.end) <= 3:
                        return True
    return False

def find_award_winner(text):
    """Attempt to extract the person who won an award based on text."""
    # Apply the NLP pipeline to the text
    doc = nlp(text)
    
    # First, find if the tweet talks about winning or awards using regex
    if re.search(win_keywords, text, re.IGNORECASE):
        
        alleged_winner = None
        subject = None
        
        # Extract named entities
        entities = extract_entities(doc)
        
        # Extract the subject (nsubj) from the sentence
        for token in doc:
            if token.dep_ == 'nsubj':
                subject = token.text
                break
        
        # Find the PERSON entity
        for ent in entities:
            if ent[1] == "PERSON":
                alleged_winner = ent[0]
                
                # Check if the subject and PERSON entity match
                if subject and is_name_match(alleged_winner, subject):
                    # Check if a win-related keyword is near the PERSON
                    if is_winner_keyword_near_person(doc, alleged_winner):
                        return f"The award winner is: {alleged_winner}"
        
        # If no clear winner keyword is near the PERSON, return possible winner
        if alleged_winner:
            return f"Possible award winner: {alleged_winner}"
    
    return "No award mention found."


# Test the function
print(find_award_winner(text))