import spacy
import re
import pandas as pd

award_show_names = [
    'GoldenGlobes', 'Golden Globes', 'Oscars', 'Academy Awards', 'Emmys',
    'Grammy Awards', 'BAFTA', 'SAG Awards', 'Tony Awards', 'Cannes Film Festival',
    'MTV Video Music Awards', 'American Music Awards', 'Critics Choice Awards',
    "People's Choice Awards", 'Billboard Music Awards', 'BET Awards',
    'Teen Choice Awards', 'Country Music Association Awards', 'Academy of Country Music Awards',
    'Golden Globe Awards', 'Emmy Awards', 'Grammy', 'Cannes', 'MTV Awards',
]

nlp = spacy.load('en_core_web_sm')
cleaned_df = pd.read_csv('text_cleaned.csv')
presenter_keywords = r'\b(presenter|presenting|presented|presents|present)\b'
presenter_data = cleaned_df[cleaned_df['text'].str.contains(presenter_keywords, case=False, na=False)]
presenter_data = presenter_data.reset_index(drop=True)

print(f"Number of tweets mentioning presenters: {len(presenter_data)}")

presenter_data.to_csv('presenter_data.csv', index=False)

def extract_person_entities(text):
    """Extract PERSON entities from text using spaCy, excluding award show names."""
    doc = nlp(text)
    persons = []
    award_show_names_lower = [name.lower() for name in award_show_names]
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            # Normalize entity text for comparison
            ent_text = ent.text.strip().lower()
            if ent_text not in award_show_names_lower:
                persons.append(ent.text)
    return persons

def extract_award_name_after_best(doc):
    """Extract the full award name starting from 'Best' using pattern matching and dependency parsing."""
    award_phrases = []
    for i, token in enumerate(doc):
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
    if award_phrases:
        extracted_award = max(award_phrases, key=len)
        award_text = extracted_award.strip().lower()
        if award_text not in [name.lower() for name in award_show_names]:
            return extracted_award
    return None

def extract_award_name_before_award(doc):
    """Extract the full award name preceding 'award' using dependency parsing."""
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
        extracted_award = max(award_phrases, key=len)
        award_text = extracted_award.strip().lower()
        if award_text not in [name.lower() for name in award_show_names]:
            return extracted_award
    return None

def extract_award_names(text):
    """Extract award names using both 'best' and 'award' triggers, excluding award show names."""
    doc = nlp(text)
    best_award = extract_award_name_after_best(doc)
    award_name = extract_award_name_before_award(doc)
    extracted_award = best_award or award_name
    if extracted_award:
        # Normalize award name for comparison
        award_text = extracted_award.strip().lower()
        award_show_names_lower = [name.lower() for name in award_show_names]
        if award_text not in award_show_names_lower:
            return extracted_award
    return None


def extract_presenter_award_pairs(text):
    """Extract presenter and award pairs, excluding award show names."""
    doc = nlp(text)
    people = extract_person_entities(text)
    award_name = extract_award_names(text)
    
    # Check if award_name is valid
    if not award_name:
        return {}
    
    # Initialize the award_presenters dictionary
    award_presenters = {}
    
    # Search for presenter keywords
    pattern = r'\b(presenting|presented|presents|present)\b'
    matches = re.finditer(pattern, text, flags=re.IGNORECASE)
    
    award_show_names_lower = [name.lower() for name in award_show_names]
    
    for match in matches:
        start, end = match.span()
        window_size = 50  # Adjust window size as needed
        left_text = text[max(0, start - window_size):start].strip()
        right_text = text[end:end + window_size].strip()

        # Match people in proximity to keywords
        for person in people:
            person_lower = person.strip().lower()
            if person_lower in award_show_names_lower:
                continue  # Skip if person is actually an award show name
            if person in left_text or person in right_text:
                if award_name in award_presenters:
                    award_presenters[award_name].add(person)
                else:
                    award_presenters[award_name] = set([person])
    # Convert sets to lists for JSON serialization
    award_presenters = {k: list(v) for k, v in award_presenters.items()}
    return award_presenters

award_show_names = [
    'GoldenGlobes', 'Golden Globes', 'Oscars', 'Academy Awards', 'Emmys',
    'Grammy Awards', 'BAFTA', 'SAG Awards', 'Tony Awards', 'Cannes Film Festival',
    'MTV Video Music Awards', 'American Music Awards', 'Critics Choice Awards',
    "People's Choice Awards", 'Billboard Music Awards', 'BET Awards',
    'Teen Choice Awards', 'Country Music Association Awards', 'Academy of Country Music Awards',
    'Golden Globe Awards', 'Emmy Awards', 'Grammy', 'Cannes', 'MTV Awards',
]

nlp = spacy.load('en_core_web_sm')
cleaned_df = pd.read_csv('text_cleaned.csv')
presenter_keywords = r'\b(presenter|presenting|presented|presents|present)\b'
presenter_data = cleaned_df[cleaned_df['text'].str.contains(presenter_keywords, case=False, na=False)]
presenter_data = presenter_data.reset_index(drop=True)

print(f"Number of tweets mentioning presenters: {len(presenter_data)}")

presenter_data.to_csv('presenter_data.csv', index=False)
# Apply entity extraction and pair extraction functions
presenter_data['Presenters'] = presenter_data['text'].apply(extract_person_entities)
presenter_data['Presenter_Award_Pairs'] = presenter_data['text'].apply(extract_presenter_award_pairs)

presenter_data = presenter_data[presenter_data['Presenter_Award_Pairs'].map(len) > 0]

def consolidate_presenters(row):
    """Consolidate presenters per award in the tweet."""
    pairs = row['Presenter_Award_Pairs']
    consolidated = []
    for award, presenters in pairs.items():
        consolidated.append({'award': award, 'presenters': presenters})
    return consolidated

presenter_data['Consolidated_Pairs'] = presenter_data.apply(consolidate_presenters, axis=1)
final_output = presenter_data[['text', 'Consolidated_Pairs']]
final_output.to_csv('presenter_award_consolidated.csv', index=False)

print("Presenter-award pair extraction completed. Results saved to 'presenter_award_consolidated.csv'.")
