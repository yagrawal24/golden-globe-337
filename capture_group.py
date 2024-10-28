import spacy
import re
import pandas as pd

# Load the SpaCy English language model
nlp = spacy.load('en_core_web_sm')

# Example tweet text
text = "rt @galaxiemag lifetime achievement award winner jodie foster is 50 she has been acting for 47 of those years amazing"
# text = "jessica chastain best actress drama winner at golden globes 2013"

# Use SpaCy's pipeline on the text
doc = nlp(text)

# Define a regex pattern to capture award-related words
# win_keywords = r"(wins|winner|awarded|received)"
# win_keywords = r"(win|wins|won|winner|victor|victory|receive|received|receives|awarded)"
win_keywords = r"(\bwin\b|\bwins\b)"

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

# def extract_award_name(text, win_match):
#     if win_match is not None:
#         words_after = text.split()[win_match + 1 : win_match + 4]
#         if "best" in words_after:
#             # Use spaCy NLP processing
#             doc = nlp(text)
            
#             # Look for noun phrases containing the word "best"
#             award_candidates = []
#             for chunk in doc.noun_chunks:
#                 # Check if the phrase contains "best" and has relevant structure
#                 if "best" in chunk.text.lower():
#                     award_candidates.append(chunk.text)

#             return award_candidates if award_candidates else None
#     return None

def extract_award_name_after_best(doc):
    """Extract the full award name starting from 'best' using dependency parsing."""
    for token in doc:
        if token.text.lower() == 'best':
            # Start from the 'best' token
            award_tokens = [token]
            # Use a list to collect tokens to the right of 'best' that are part of the award name
            for right_token in doc[token.i + 1:]:
                # Stop if we reach a punctuation mark or a coordinating conjunction
                if right_token.pos_ in ('PUNCT', 'CCONJ', 'VERB') or right_token.dep_ == 'punct':
                    break
                award_tokens.append(right_token)
            # Combine the tokens to form the award name
            award_name = ' '.join([t.text for t in award_tokens])
            return award_name
    return None

def find_award_winner(text):
    """Attempt to extract the person who won an award based on text."""
    # Apply the NLP pipeline to the text
    doc = nlp(text)
    
    # First, find if the tweet talks about winning or awards using regex
    if re.search(win_keywords, text, re.IGNORECASE):

        # win_match = next((index for index, word in enumerate(text.split()) if re.search(win_keywords, word, re.IGNORECASE)), None)
        
        alleged_winner = None
        subject = None
        award = ""
        confident_pred = False
        
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
                        confident_pred = True
                        # return f"The award winner is: {alleged_winner}"
        
        # # If no clear winner keyword is near the PERSON, return possible winner
        # if alleged_winner:
        #     return f"Possible award winner: {alleged_winner}"

        # Getting award name
        # award = extract_award_name(text, win_match)
        award = extract_award_name_after_best(doc)
        
        if award != None:
            return f"{alleged_winner} | {award}"

        # return award (?:(win|wins)\s+
        word_list = ["award", "prize", "honor", "medal", "trophy"]
        pattern = r'wins\s+(.*?\b(?:' + '|'.join(word_list) + r')\b)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            award_name = match.group(1)
            return f"{alleged_winner} | {award_name}"
    
    return None

# Test the function
win_data = pd.read_csv('wins.csv')['text'].tail(1000)
# print(win_data)
output = win_data.apply(lambda x: find_award_winner(x))

output.to_csv('award_names.csv')

# output = win_data.head(500).apply(lambda x: (x, find_award_winner(x)))
# output_df = pd.DataFrame(output.tolist(), columns=["text", "extracted_award"])

# Save the DataFrame to a CSV file
# output_df.to_csv('award_names_with_text.csv', index=False)