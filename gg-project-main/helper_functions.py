#############################################################################
# ALL EXPLANATIONS TO FUNCTIONS AND CODE PROVIDED IN final_submission.ipynb #
#############################################################################
import pandas as pd
import numpy as np
import re
import spacy
from collections import Counter
from collections import defaultdict

import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_data():
    df = pd.read_json('gg2013.json')['text']
    def clean(text):
        if re.search(r'[^\x00-\x7F\u263a-\U0001f645]', text): 
            return None
        text = re.sub(r'http\S+|www\S+|pic.twitter\S+', '', text)
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]', '', text)
        text = re.sub(r' +', ' ', text).strip()
        return text

    df = df.apply(clean)
    cleaned_data = df.dropna()
    cleaned_data = cleaned_data[cleaned_data.str.strip() != ""]
    cleaned_data.to_csv('text_cleaned.csv', index=False)
    return cleaned_data

def extract_entities_as_nominee(doc):
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'WORK_OF_ART', 'ORG', 'PRODUCT']:
            return ent.text
    return None

def extract_full_subject_as_nominee(doc):
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
                if next_token.text.lower() == 'for':
                    break
                award_tokens.append(next_token)
            award_phrase = ' '.join([t.text for t in award_tokens]).strip()
            if award_phrase:
                award_phrases.append(award_phrase)
    if award_phrases:
        return max(award_phrases, key=len)
    return None

def extract_award_name_before_award(doc):
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

def extract_award_names(text, nlp, award_show_names):
    doc = nlp(text)
    best_award = extract_award_name_after_best(doc)
    award_name = extract_award_name_before_award(doc)
    extracted_award = best_award or award_name
    if extracted_award:
        award_text = extracted_award.strip().lower()
        award_show_names_lower = [name.lower() for name in award_show_names]
        if award_text not in award_show_names_lower:
            return extracted_award
    return None

def ignore_rt_and_mentions(text, nlp):
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not (token.text.lower() == 'rt' or token.text.startswith('@'))]
    return ' '.join(filtered_tokens)

def find_award_winner(text, nlp, win_keywords, award_show_names):
    filtered_text = ignore_rt_and_mentions(text, nlp)
    doc = nlp(filtered_text)

    if re.search(win_keywords, filtered_text, re.IGNORECASE):
        nominee = extract_full_subject_as_nominee(doc)
        if not nominee:
            nominee = extract_entities_as_nominee(doc)

        award_category = extract_award_names(doc, nlp, award_show_names)
        
        if award_category != None and nominee != None:
            if re.search(r"(win|#|@)", award_category, re.IGNORECASE) != None:
                return None
            if re.search(r"(win|#|@)", nominee, re.IGNORECASE) != None:
                return None
    
            return {award_category: nominee}
    
    return None

def help_get_winners():
    nlp = spacy.load('en_core_web_sm')
    win_keywords = r"(\bwins\b)"
    award_show_names = [
        'GoldenGlobes', 'Golden Globes', 'Oscars', 'Academy Awards', 'Emmys',
        'Grammy Awards', 'BAFTA', 'SAG Awards', 'Tony Awards', 'Cannes Film Festival',
        'MTV Video Music Awards', 'American Music Awards', 'Critics Choice Awards',
        "People's Choice Awards", 'Billboard Music Awards', 'BET Awards',
        'Teen Choice Awards', 'Country Music Association Awards', 'Academy of Country Music Awards',
        'Golden Globe Awards', 'Emmy Awards', 'Grammy', 'Cannes', 'MTV Awards',
    ]

    cleaned_data = pd.read_csv('text_cleaned.csv')['text']
    win_data = cleaned_data[cleaned_data.apply(lambda x: re.search(win_keywords, x) != None)]
    win_output = win_data.apply(find_award_winner, args=(nlp, win_keywords, award_show_names))
    win_output = win_output.dropna()
    # win_output.to_csv('winners_and_awards.csv')
    return win_output

def extract_person_entities(text, nlp):
    """Extract PERSON entities while excluding Twitter handles and hashtags."""
    doc = nlp(text)
    persons = []
    for ent in doc.ents:
        if ent.label_ == 'PERSON' and not (ent.text.startswith('@') or ent.text.startswith('#')):
            persons.append(ent.text)
    return persons

def infer_award_names(text, nlp):
    """Extract potential award phrases using common descriptors and categories."""
    descriptors = ["Best", "Outstanding", "Top", "Achievement in", "Excellence in"]
    categories = [
        "Actor", "Actress", "Director", "Picture", "Screenplay", "Soundtrack",
        "Album", "Song", "Artist", "Performance", "Music Video", "Television Series",
        "Drama", "Comedy", "Animated", "Documentary", "Feature Film", "Reality Show",
        "Supporting Actor", "Supporting Actress"
    ]
    doc = nlp(text)
    for desc in descriptors:
        for cat in categories:
            pattern = rf"{desc}.*{cat}"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)  # Return the matching phrase
    return None

# Match awards and presenters in a sentence based on common patterns
def extract_presenter_award_pairs(text, nlp, presenter_keywords):
    """Extract presenters associated with inferred awards in a sentence."""
    doc = nlp(text)
    people = extract_person_entities(text, nlp)
    award_name = infer_award_names(text, nlp)
    award_presenters = defaultdict(set)

    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in presenter_keywords):
            for person in people:
                if person.lower() in sent.text.lower():
                    award_presenters[award_name].add(person)
    
    return {award: list(presenters) for award, presenters in award_presenters.items() if presenters}

# Consolidate presenters per award entry
def consolidate_presenters(row):
    presenter_award_pairs = row['Presenter_Award_Pairs']
    consolidated = defaultdict(set)

    for award, presenters in presenter_award_pairs.items():
        consolidated[award].update(presenters)

    return {award: list(presenters) for award, presenters in consolidated.items()}

def help_get_presenters():
    nlp = spacy.load('en_core_web_sm')
    presenter_keywords = r'\b(presenter|presenting|presented|presents|present)\b'

    award_show_names = [
        'GoldenGlobes', 'Golden Globes', 'Oscars', 'Academy Awards', 'Emmys',
        'Grammy Awards', 'BAFTA', 'SAG Awards', 'Tony Awards', 'Cannes Film Festival',
        'MTV Video Music Awards', 'American Music Awards', 'Critics Choice Awards',
        "People's Choice Awards", 'Billboard Music Awards', 'BET Awards',
        'Teen Choice Awards', 'Country Music Association Awards', 'Academy of Country Music Awards',
        'Golden Globe Awards', 'Emmy Awards', 'Grammy', 'Cannes', 'MTV Awards',
    ]

    cleaned_df = pd.read_csv('text_cleaned.csv')
    presenter_data = cleaned_df[cleaned_df['text'].str.contains(presenter_keywords, case=False, na=False)].reset_index(drop=True)
    print(f"Filtered data to {len(presenter_data)} rows with potential presenter mentions.")
    presenter_data['Presenters'] = presenter_data['text'].apply(extract_person_entities, args=[nlp])
    presenter_data['Presenter_Award_Pairs'] = presenter_data['text'].apply(extract_presenter_award_pairs, args=[nlp, presenter_keywords])
    
    presenter_data = presenter_data[presenter_data['Presenter_Award_Pairs'].map(len) > 0]
    print(f"Found {len(presenter_data)} rows with valid presenter-award pairs.")

    # Consolidate presenters per award
    presenter_data['Consolidated_Pairs'] = presenter_data.apply(consolidate_presenters, axis=1)

    # Print consolidated output directly
    for idx, row in presenter_data['Consolidated_Pairs'].items():
        print(row)

    return presenter_data['Consolidated_Pairs']


def convert_results_to_match_awards(awards, win_output):
    d2 = {award: None for award in awards}
    input_list = win_output

    d1 = {k: v for item in input_list for k, v in item.items()}

    all_keys = list(d2.keys()) + list(d1.keys())
    vectorizer = TfidfVectorizer().fit(all_keys)
    award_vectors = vectorizer.transform(list(d2.keys())) 
    d1_vectors = vectorizer.transform(list(d1.keys()))

    similarity_matrix = cosine_similarity(award_vectors, d1_vectors)

    best_match_indices = np.argmax(similarity_matrix, axis=1)
    for idx, award in enumerate(d2.keys()):
        best_match_key = list(d1.keys())[best_match_indices[idx]]
        d2[award] = d1[best_match_key]

    return d2