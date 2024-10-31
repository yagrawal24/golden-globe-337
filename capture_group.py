import spacy
import re
import pandas as pd
import json  # Ensure json is imported

# Load the SpaCy English language model
print("Loading SpaCy model...")
nlp = spacy.load('en_core_web_sm')
print("SpaCy model loaded successfully.")

# Define regex patterns to capture award-related keywords
win_keywords = r"(\bwin\b|\bwins\b|\bwon\b|\bawarded\b)"
nominee_keywords = r"(\bnominated\b|\bnominee\b|\bnomination\b|\bhope\b|\bwish\b|\bthink\b|\bbelieve\b|\bshould\b)"

def extract_entities_as_nominee(doc):
    """Extract named entities that could serve as a nominee/winner."""
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'WORK_OF_ART', 'ORG', 'PRODUCT']:
            return ent.text
    return None

def extract_full_subject_as_nominee(doc):
    """Extract the full subject (nsubj) and its modifiers (compound, det)."""
    for token in doc:
        if token.dep_ == 'nsubj' and token.head.text in ['wins', 'won', 'receives']:
            subject_tokens = []
            for left in token.lefts:
                if left.dep_ in ['det', 'compound']:
                    subject_tokens.append(left.text)
            subject_tokens.append(token.text)
            return ' '.join(subject_tokens)
    return None

def extract_award_name_after_best(doc):
    """Extract the full award name starting from 'Best'."""
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
    """Extract the full award name preceding 'award'."""
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
    return best_award or award_name

def ignore_rt_and_mentions(text):
    """Ignore 'rt' and '@' usernames but continue parsing the rest."""
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not (token.text.lower() == 'rt' or token.text.startswith('@'))]
    return ' '.join(filtered_tokens)

def find_award_winner(text):
    """Attempt to extract award information and return a structured output."""
    filtered_text = ignore_rt_and_mentions(text)
    doc = nlp(filtered_text)

    if re.search(win_keywords, filtered_text, re.IGNORECASE):
        nominee = extract_full_subject_as_nominee(doc) or extract_entities_as_nominee(doc)
        award_category = extract_award_names(doc)

        hope_regex = "hope|wish|think|believe|should"
        nominee_match = re.search(hope_regex, text, re.IGNORECASE)

        if award_category:
            role = "nominee" if nominee_match else "winner"
            return f"{nominee} | {award_category} | {role}"

        word_list = ["award", "prize", "honor", "medal", "trophy"]
        pattern = r'wins\s+(.*?\b(?:' + '|'.join(word_list) + r')\b)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            award_category = match.group(1)
            role = "nominee" if nominee_match else "winner"
            return f"{nominee} | {award_category} | {role}"
    
    return None

def find_nominee(text):
    """Extract nominee information."""
    filtered_text = ignore_rt_and_mentions(text)
    doc = nlp(filtered_text)

    if re.search(nominee_keywords, filtered_text, re.IGNORECASE):
        nominee = extract_full_subject_as_nominee(doc) or extract_entities_as_nominee(doc)
        award_category = extract_award_name_after_best(doc)
        if nominee and award_category:
            return f"{nominee} | {award_category} | nominee"
    return None

if __name__ == "__main__":
    print("Starting test cases...")
    text1 = 'game change wins best miniseries or tv movie'
    text2 = 'i hope the hour wins best miniseries'
    text3 = 'rt @perezhilton @benaffleck argo wins best drama at the golden globes'

    # Testing outputs
    print("Test output for text1:", find_award_winner(text1))
    print("Test output for text2:", find_award_winner(text2))
    print("Test output for text3:", find_award_winner(text3))

    # Load the dataset and apply the function
    print("Loading CSV files...")
    win_data = pd.read_csv('wins.csv')['text']
    nominee_data = pd.read_csv('nominees.csv')['text']

    print("Processing winners...")
    win_output = win_data.apply(find_award_winner)
    print("Winners processed.")

    # Uncomment if nominees need processing
    # print("Processing nominees...")
    # nom_output = nominee_data.apply(find_nominee)
    # print("Nominees processed.")

    # Save the output
    print("Saving outputs...")
    win_output.to_csv('winners_and_awards.csv')
    # nom_output.to_csv('nominees_output.csv')  # Uncomment if nominees need saving
    print("Output files saved.")
