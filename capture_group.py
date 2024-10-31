import spacy
import re
import pandas as pd

# Load the SpaCy English language model
nlp = spacy.load('en_core_web_sm')

# Define regex pattern to capture award-related keywords
win_keywords = r"(\bwin\b|\bwins\b|\bwon\b|\bawarded\b)"

nominee_keywords = r"(\bnominated\b|\bnominee\b|\bnomination\b|\bhope\b|\bwish\b|\bthink\b|\bbelieve\b|\bshould\b)"

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
    """Extract the full award name starting from 'Best' using pattern matching and dependency parsing."""
    award_phrases = []
    for i, token in enumerate(doc):
        if token.text.lower() == 'best':
            award_tokens = [token]
            for j in range(i + 1, len(doc)):
                next_token = doc[j]
                if next_token.text in ('.', ',', ':', ';', '!', '?', '-', 'RT', '@', '#') or next_token.dep_ == 'punct':
                    break
                if next_token.pos_ in ('VERB', 'AUX') and next_token.dep_ in ('ROOT', 'conj'):
                    break
                award_tokens.append(next_token)
            award_phrase = ' '.join([t.text for t in award_tokens]).strip()
            if award_phrase:
                award_phrases.append(award_phrase)
    if award_phrases:
        return max(award_phrases, key=len)
    return None

def extract_award_name_before_award(doc):
    """Extract the full award name preceding 'award' using pattern matching and dependency parsing."""
    award_phrases = []
    for i, token in enumerate(doc):
        if token.text.lower() == 'award':
            award_tokens = []
            for left_token in reversed(doc[:i]):
                if left_token.text in ('.', ',', ':', ';', '!', '?', '-', 'RT', '@', '#') or left_token.dep_ == 'punct':
                    break
                if left_token.pos_ in ('VERB', 'AUX') and left_token.dep_ in ('ROOT', 'conj'):
                    break
                award_tokens.insert(0, left_token)
            award_phrase = ' '.join([t.text for t in award_tokens]).strip()
            if award_phrase:
                award_phrases.append(award_phrase)
    if award_phrases:
        return max(award_phrases, key=len)
    return None

def extract_award_names(text):
    """Extract award names using both 'best' and 'award' triggers."""
    doc = nlp(text)
    best_award = extract_award_name_after_best(doc)
    award_name = extract_award_name_before_award(doc)
    return best_award or award_name  # Return whichever is found first, preferring 'best' awards

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
        award_category = extract_award_names(doc)
        
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

def find_nominee(text):
    """Extract nominee information."""
    filtered_text = ignore_rt_and_mentions(text)
    doc = nlp(filtered_text)
    
    if re.search(nominee_keywords, filtered_text, re.IGNORECASE):
        nominee = extract_full_subject_as_nominee(doc)
        if not nominee:
            nominee = extract_entities_as_nominee(doc)
        award_category = extract_award_name_after_best(doc)
        if nominee and award_category:
            return f"{nominee} | {award_category} | nominee"
    return None



if __name__ == "__main__":
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
    nominee_data = pd.read_csv('nominees.csv')['text']

    win_output = win_data.apply(find_award_winner)
    # nom_output = nominee_data.apply(find_nominee)

    # Create a DataFrame with "Tweet" and "Output" columns
    # output = pd.concat([win_output, nom_output])

    # Save the output
    # output.to_csv('winners_nominees.csv', index=False)
    win_output.to_csv('winners_and_awards.csv')
