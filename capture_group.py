import spacy
import re
import pandas as pd

# Load the SpaCy English language model
nlp = spacy.load('en_core_web_sm')

# Define regex pattern to capture award-related keywords
win_keywords = r"(\bwin\b|\bwins\b|\bwon\b|\bawarded\b)"
nominee_keywords = r"\bnominated\b|\bnominee\b|\bnomination\b|\bhope\b|\bwish\b|\bthink\b|\bbelieve\b|\bshould\b"

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
    """Extract award information for winners."""
    filtered_text = ignore_rt_and_mentions(text)
    doc = nlp(filtered_text)
    
    if re.search(win_keywords, filtered_text, re.IGNORECASE):
        nominee = extract_full_subject_as_nominee(doc)
        if not nominee:
            nominee = extract_entities_as_nominee(doc)
        award_category = extract_award_name_after_best(doc)
        if nominee and award_category:
            return {'winner': nominee, 'award_category': award_category}
    return {'winner': None, 'award_category': None}

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
            return {'nominee': nominee, 'award_category': award_category}
    return {'nominee': None, 'award_category': None}

# Step 1: Load the wins data and extract winners
win_data = pd.read_csv('wins.csv')['text']
win_output = win_data.apply(lambda x: find_award_winner(x))
winners_df = pd.DataFrame(win_output.tolist())
winners_df['text'] = win_data

# Step 2: Load the nominees data and extract nominees
nominee_data = pd.read_csv('nominees.csv')['text']
nominee_output = nominee_data.apply(lambda x: find_nominee(x))
nominees_df = pd.DataFrame(nominee_output.tolist())
nominees_df['text'] = nominee_data

# Step 3: Match nominees to the same award categories from wins.csv
# Merge winners and nominees based on the award category
merged_df = pd.merge(winners_df, nominees_df, on='award_category', how='left', suffixes=('_winner', '_nominee'))

# Group nominees under the same award
grouped_df = merged_df.groupby('award_category').agg({
    'winner': 'first',  # Keep the first winner (since there's only one winner per award)
    'nominee': lambda x: ', '.join(filter(None, x.astype(str)))  # Ensure nominee is a string and combine all nominees for the same award
}).reset_index()

# Step 4: Format the output with winners and nominees
grouped_df['output'] = grouped_df.apply(
    lambda row: f"Winner: {row['winner']}, Nominees: {row['nominee']}, Award Category: {row['award_category']}",
    axis=1
)

# Step 5: Save the final output to a CSV
grouped_df[['award_category', 'output']].to_csv('winners_and_nominees.csv', index=False)
