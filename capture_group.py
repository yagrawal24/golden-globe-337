import spacy
import re
import pandas as pd

# Load the SpaCy English language model
nlp = spacy.load('en_core_web_sm')

# Define regex pattern to capture award-related keywords
win_keywords = r"(\bwin\b|\bwins\b|\bwon\b|\bawarded\b)"

def extract_entities_as_nominee(doc):
    """Extract named entities that could serve as a nominee/winner."""
    for ent in doc.ents:
        # Consider entities such as PERSON, WORK_OF_ART, ORG, PRODUCT (e.g., "Argo")
        if ent.label_ in ['PERSON', 'WORK_OF_ART', 'ORG', 'PRODUCT']:
            return ent.text
    return None

def extract_full_subject_as_nominee(doc):
    """Extract the full subject (nsubj) and its modifiers (compound, det)."""
    for token in doc:
        if token.dep_ == 'nsubj' and token.head.text in ['wins', 'won', 'receives']:
            subject_tokens = []
            # Collect compound tokens (like "game" in "game change")
            for left in token.lefts:
                if left.dep_ in ['det', 'compound']:  # Determiner or compound part
                    subject_tokens.append(left.text)
            subject_tokens.append(token.text)  # Add the main subject token
            return ' '.join(subject_tokens)  # Return full subject as a string
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

def ignore_rt_and_mentions(text):
    """Ignore 'rt' and '@' usernames but continue parsing the rest."""
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not (token.text.lower() == 'rt' or token.text.startswith('@'))]
    return ' '.join(filtered_tokens)

def find_award_winner(text):
    """Attempt to extract award information and return a structured output."""
    
    # Ignore 'rt' and mentions but continue with the rest of the tweet
    filtered_text = ignore_rt_and_mentions(text)
    
    doc = nlp(filtered_text)
    
    # Check if the tweet mentions winning or awards
    if re.search(win_keywords, filtered_text, re.IGNORECASE):
        # Extract the nominee (winner)
        nominee = extract_full_subject_as_nominee(doc)
        if not nominee:
            nominee = extract_entities_as_nominee(doc)

        # Extract the award category
        award_category = extract_award_name_after_best(doc)
        
        hope_regex = "hope|wish|think|believe|should"
        nominee_match = re.search(hope_regex, text, re.IGNORECASE)
        
        if award_category != None:
            if nominee_match != None:
                return f"{nominee} | {award_category} | nominee"
            return f"{nominee} | {award_category} | winner"

        # return award (?:(win|wins)\s+
        word_list = ["award", "prize", "honor", "medal", "trophy"]
        pattern = r'wins\s+(.*?\b(?:' + '|'.join(word_list) + r')\b)'
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            award_category = match.group(1)
            if nominee_match != None:
                return f"{nominee} | {award_category} | nominee"
            return f"{nominee} | {award_category} | winner"
    
    return None


# Test the function with multiple test cases
text1 = 'game change wins best miniseries or tv movie'
text2 = 'i hope the hour wins best miniseries'
text3 = 'rt @perezhilton @benaffleck argo wins best drama at the golden globes'

# Testing outputs
print(find_award_winner(text1))  # 'Winner: game change, Award Category: best miniseries or tv movie'
print(find_award_winner(text2))  # 'Winner: the hour, Award Category: best miniseries'
print(find_award_winner(text3))  # 'Winner: argo, Award Category: best drama'

# Load the dataset and apply the function
win_data = pd.read_csv('wins.csv')['text']

output = win_data.apply(lambda x: find_award_winner(x))

# Create a DataFrame with "Tweet" and "Output" columns
output_df = pd.DataFrame({
    'Tweet': win_data,
    'Output': output
})

# Save the output
output_df.to_csv('award_names.csv', index=False)
