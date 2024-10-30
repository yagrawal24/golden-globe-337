import spacy
import re
import pandas as pd

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the cleaned dataset as a DataFrame
cleaned_df = pd.read_csv('text_cleaned.csv')

# Define presenter-related keywords pattern
presenter_keywords = r'\b(presenter|presenting|presented|presents|present)\b'

# Filter rows in 'text' column containing presenter-related keywords
presenter_data = cleaned_df[cleaned_df['text'].str.contains(presenter_keywords, case=False, na=False)]

# Optional: Reset index for easier handling
presenter_data = presenter_data.reset_index(drop=True)

# Output the count of tweets mentioning presenters
print(f"Number of tweets mentioning presenters: {len(presenter_data)}")

# Save the filtered data to a CSV file
presenter_data.to_csv('presenter_data.csv', index=False)

def extract_person_entities(text):
    """Extract PERSON entities from text using spaCy."""
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    return persons

def extract_award_name_after_best(doc):
    """Extract the full award name starting from 'Best' using pattern matching and dependency parsing."""
    award_phrases = []
    for i, token in enumerate(doc):
        # Look for 'Best' (case-insensitive)
        if token.text.lower() == 'best':
            award_tokens = [token]
            # Collect tokens that likely belong to the award name
            for j in range(i + 1, len(doc)):
                next_token = doc[j]
                # Stop if we hit punctuation that likely ends the award name
                if next_token.text in ('.', ',', ':', ';', '!', '?', '-', 'RT', '@', '#') or next_token.dep_ == 'punct':
                    break
                # Stop if we hit verbs that likely indicate the start of a new clause
                if next_token.pos_ in ('VERB', 'AUX') and next_token.dep_ in ('ROOT', 'conj'):
                    break
                award_tokens.append(next_token)
            award_phrase = ' '.join([t.text for t in award_tokens])
            # Clean up any trailing conjunctions or prepositions
            award_phrase = award_phrase.strip()
            if award_phrase:
                award_phrases.append(award_phrase)
    # Return the longest award phrase found
    if award_phrases:
        return max(award_phrases, key=len)
    return None


def extract_award_name_before_award(doc):
    """Extract the full award name preceding 'award' using dependency parsing."""
    for token in doc:
        if token.text.lower() == 'award':
            award_tokens = []
            for left_token in reversed(doc[:token.i]):
                if left_token.pos_ in ('PUNCT', 'CCONJ', 'VERB', 'ADP', 'DET') or left_token.dep_ == 'punct':
                    break
                award_tokens.insert(0, left_token)
            return ' '.join([t.text for t in award_tokens])
    return None

def extract_award_names(text):
    """Extract award names using both 'best' and 'award' triggers."""
    doc = nlp(text)
    best_award = extract_award_name_after_best(doc)
    award_name = extract_award_name_before_award(doc)
    return best_award or award_name

def extract_presenter_award_pairs(text):
    """Extract presenter and award pairs based on specific patterns."""
    doc = nlp(text)
    people = extract_person_entities(text)
    award_name = extract_award_names(text)
    
    pairs = []
    if award_name:
        # Search for presenter keywords
        pattern = r'\b(presenting|presented|presents|present)\b'
        matches = re.finditer(pattern, text, flags=re.IGNORECASE)
        
        for match in matches:
            start, end = match.span()
            left_text = text[:start].strip()
            right_text = text[end:].strip()

            # Match people and award in proximity to keywords
            for person in people:
                if person in left_text or person in right_text:
                    pairs.append((person, award_name))
    return pairs

# Apply entity extraction and pair extraction functions
presenter_data['Presenters'] = presenter_data['text'].apply(extract_person_entities)
presenter_data['Presenter_Award_Pairs'] = presenter_data['text'].apply(extract_presenter_award_pairs)

# Filter out rows without presenter-award pairs
presenter_data = presenter_data[presenter_data['Presenter_Award_Pairs'].map(len) > 0]

# Flatten the list of pairs for final output
all_pairs = presenter_data.explode('Presenter_Award_Pairs')[['text', 'Presenter_Award_Pairs']]

# Save the presenter-award pairs to a CSV file
all_pairs.to_csv('presenter_award_pairs.csv', index=False)

print("Presenter-award pair extraction completed. Results saved to 'presenter_award_pairs.csv'.")
