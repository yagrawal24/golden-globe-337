#############################################################################
# ALL EXPLANATIONS TO FUNCTIONS AND CODE PROVIDED IN final_submission.ipynb #
#############################################################################
import pandas as pd
import numpy as np
import re
import spacy
from collections import Counter
from collections import defaultdict
import wikipediaapi

import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")
joke_phrases = ["best joke", "funniest joke", "best comedian", "funniest moment"]

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
    # NEED TO FIND WAY TO REMOVE NAMES, MOVIES FROM AWARD NAMES
    # EITHER FOUND WITH DOC.ENTS OR AFTER "-" IN DOC.ENTS
    
    award_phrases = []
    punct_count = 0
    
    for i, token in enumerate(doc):
        if token.text.lower() == 'best':
            award_tokens = [token]
            for j in range(i + 1, len(doc)):
                next_token = doc[j]
                if next_token.dep_ == 'punct':
                    punct_count += 1
                if punct_count >= 2:
                    break
                if next_token.text in ('.', ',', ':', ';', '!', '?', 'RT', '@', '#'):# or next_token.dep_ == 'punct':
                    break
                if next_token.pos_ in ('VERB', 'AUX') and next_token.dep_ in ('ROOT', 'conj'):
                    break
                if next_token.text.lower() in ['for','win', 'won', 'by', 'goes']:# or [ent.find(next_token.text.lower()) for ent in entities]:
                    break
                if j+1 < len(doc) and next_token.dep_ == 'punct' and (doc[j+1].dep_ == 'compound' or doc[j+1].dep_ == 'punct'):
                    break
                award_tokens.append(next_token)
            award_phrase = ' '.join([t.text for t in award_tokens]).strip()
            if award_phrase:
                award_phrases.append(award_phrase)
    if award_phrases:
        return max(award_phrases, key=len)
    return None

def extract_award_name_before_award(doc):
    award_list = ['award', 'honor', 'prize', 'trophy']

    award_name = ""
    prev_type = ""
    for i, token in enumerate(reversed(doc)):
        if str(token) in award_list:
            award_name += " " + str(token)
            prev_type = "dobj"
        elif prev_type == "dobj" and token.dep_ == "compound":
            award_name += " " + str(token)
        else:
            break

    if len(award_name) > 0:
        a = award_name.split(" ")
        a.reverse()
        return ' '.join(a)
    return None
# def extract_award_name_before_award(doc):
#     award_phrases = []
#     for i, token in enumerate(doc):
#         if token.text.lower() == 'award':
#             award_tokens = []
#             for left_token in reversed(doc[:i]):
#                 if left_token.text in ('.', ',', ':', ';', '!', '?', '-', 'RT', '@', '#') or left_token.dep_ == 'punct':
#                     break
#                 if left_token.pos_ in ('VERB', 'AUX') and left_token.dep_ in ('ROOT', 'conj'):
#                     break
#                 award_tokens.insert(0, left_token)
#             award_phrase = ' '.join([t.text for t in award_tokens]).strip()
#             if award_phrase:
#                 award_phrases.append(award_phrase)
#     if award_phrases:
#         return max(award_phrases, key=len)
#     return None

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

def help_get_awards():
    from itertools import islice
    
    nlp = spacy.load('en_core_web_sm')
    award_keywords = r"(\bbest\b|\baward\b|\boutstanding\b|\bfavorite\b|\bfavourite\b|\btop\b|\bhonor\b|\bprize\b|\bchoice\b)"
    award_show_names = [
        'GoldenGlobes', 'Golden Globes', 'Oscars', 'Academy Awards', 'Emmys',
        'Grammy Awards', 'BAFTA', 'SAG Awards', 'Tony Awards', 'Cannes Film Festival',
        'MTV Video Music Awards', 'American Music Awards', 'Critics Choice Awards',
        "People's Choice Awards", 'Billboard Music Awards', 'BET Awards',
        'Teen Choice Awards', 'Country Music Association Awards', 'Academy of Country Music Awards',
        'Golden Globe Awards', 'Emmy Awards', 'Grammy', 'Cannes', 'MTV Awards',
    ]
    cleaned_data = pd.read_csv('text_cleaned.csv')['text']
    
    account_rts = {}

    for text in cleaned_data:
        tweet = text.split(' ')
        if tweet[0] == 'RT':
            if tweet[1] in account_rts:
                account_rts[tweet[1]] += 1
            else:
                account_rts.update({tweet[1]:1})

    account_dicts = dict(sorted(account_rts.items(), key=lambda item: item[1], reverse=True))

    top_accounts = list(islice(account_dicts, 20))
    
    retweet_pattern = r'RT\s' + '|'.join(top_accounts)
    reputable_df = cleaned_data.apply(lambda text: text if re.search(retweet_pattern, text, re.IGNORECASE) else None)
    reputable_df.dropna(inplace=True)
    
    award_data = reputable_df.apply(lambda x: x if re.search(award_keywords, x.lower()) != None else None)
    award_data.dropna(inplace=True)
    
    award_data = award_data.apply(lambda x: extract_award_names(x, nlp, award_show_names))
    award_data = award_data.dropna().apply(lambda x: x.lower()).unique()
    
    print(len(award_data))
    
    return award_data

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

def extract_presenter_award_pairs(text, nlp, award_show_names):
    """Extract presenters associated with inferred awards in a sentence."""
    people = extract_person_entities(text, nlp)
    award_name = extract_award_names(text, nlp, award_show_names)

    if award_name != None and people != []:
        return {award_name: people}
    else:
        return None

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
    # presenter_data = cleaned_df[cleaned_df['text'].str.contains(presenter_keywords, case=False, na=False)].reset_index(drop=True)
    presenter_data = cleaned_df[cleaned_df['text'].str.extract(f'({presenter_keywords})', flags=re.IGNORECASE).notnull().any(axis=1)].copy()

    presenter_data['Presenter_Award_Pairs'] = presenter_data['text'].apply(extract_presenter_award_pairs, args=[nlp, award_show_names])
    presenter_data = presenter_data.dropna(subset=['Presenter_Award_Pairs'])
    presenter_data['Consolidated_Pairs'] = presenter_data.apply(consolidate_presenters, axis=1)

    return presenter_data['Consolidated_Pairs']

def help_get_hosts():
    award_show_names = [
        'GoldenGlobes', 'Golden Globes', 'Oscars', 'Academy Awards', 'Emmys',
        'Grammy Awards', 'BAFTA', 'SAG Awards', 'Tony Awards', 'Cannes Film Festival',
        'MTV Video Music Awards', 'American Music Awards', 'Critics Choice Awards',
        "People's Choice Awards", 'Billboard Music Awards', 'BET Awards',
        'Teen Choice Awards', 'Country Music Association Awards', 'Academy of Country Music Awards',
        'Golden Globe Awards', 'Emmy Awards', 'Grammy', 'Cannes', 'MTV Awards',
    ]
    wiki = wikipediaapi.Wikipedia('student_homework')
    cleaned_data = pd.read_csv('text_cleaned.csv')['text']
    nlp = spacy.load('en_core_web_sm')

    def lookup_wikipedia(name):
        page = wiki.page(name)
        if page.exists():
            return re.search("born", page.summary, re.IGNORECASE) != None 
        else:
            return False
        
    host_keywords = r'\b(host|hosts|hosting)\b'

    # host_data = cleaned_data[
    #     cleaned_data.str.contains(host_keywords, case=False, na=False)
    # ]

    host_data = cleaned_data[cleaned_data.str.extract(f'({host_keywords})', flags=re.IGNORECASE).notnull().any(axis=1)].copy()

    early_host_data = host_data[:int(0.1*len(host_data))].apply(extract_person_entities, args=[nlp])
    all_names = [name for names_list in early_host_data for name in names_list]
    name_counts = Counter(all_names)

    hosts = []
    i = 0
    potential = name_counts.most_common()

    filtered_potential = [
        (name, count) for name, count in potential 
        if name.lower() not in [award.lower() for award in award_show_names] and lookup_wikipedia(name)
    ]

    while len(hosts) < 2 and len(hosts) < len(filtered_potential):
        curr = filtered_potential[i][0]
        if lookup_wikipedia(curr):
            hosts.append(curr)
        i += 1

    return hosts

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

def find_nominees(text, nlp, nom_keywords, award_show_names):
    filtered_text = ignore_rt_and_mentions(text, nlp)
    doc = nlp(filtered_text)

    if re.search(nom_keywords, filtered_text, re.IGNORECASE):
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

def help_get_nominees():
    nlp = spacy.load('en_core_web_sm')
    nominee_keywords = r"(\bnominee\b|\bnominate\b|\bnominated\b)"
    award_show_names = [
        'GoldenGlobes', 'Golden Globes', 'Oscars', 'Academy Awards', 'Emmys',
        'Grammy Awards', 'BAFTA', 'SAG Awards', 'Tony Awards', 'Cannes Film Festival',
        'MTV Video Music Awards', 'American Music Awards', 'Critics Choice Awards',
        "People's Choice Awards", 'Billboard Music Awards', 'BET Awards',
        'Teen Choice Awards', 'Country Music Association Awards', 'Academy of Country Music Awards',
        'Golden Globe Awards', 'Emmy Awards', 'Grammy', 'Cannes', 'MTV Awards',
    ]

    cleaned_data = pd.read_csv('text_cleaned.csv')['text']
    nom_data = cleaned_data[cleaned_data.apply(lambda x: re.search(nominee_keywords, x) != None)]
    nom_output = nom_data.apply(find_nominees, args=(nlp, nominee_keywords, award_show_names))
    nom_output = nom_output.dropna()
    # nom_output.to_csv('nominees.csv')
    return nom_output

# def get_extra_credit():


def human_readable_version(award_names):
    cleaned_data = clean_data()
    hosts = help_get_hosts()
    award_list = help_get_awards()
    winners = help_get_winners()
    presenters = help_get_presenters()
    nominees = help_get_nominees()

    extra_credit = get_best_dressed_and_jokes()

    text_winners = convert_results_to_match_awards(award_names, winners)
    text_presenters = convert_results_to_match_awards(award_names, presenters)
    text_nominees = convert_results_to_match_awards(award_names, nominees)

    output = ""
    output += f"Host: {', '.join(hosts)}\n\n"

    for award in award_names:
        output += f"Award: {award}\n"
        
        # Add presenters, nominees, and winner for each award
        presenters = text_presenters.get(award, [])
        nominees = text_nominees.get(award, [])
        winner = text_winners.get(award, "")

        output += f"Presenters: {', '.join(presenters)}\n"
        # output += f"Nominees: {', '.join(nominees)}\n"
        output += f"Nominees: {nominees}\n\n" # RIGHT NOW FOR BAD NOMINEES, CHANGE TO LIST VERSION WHEN NOMINEES IS A LIST
        output += f"Winner: {winner}\n\n"
    
    # Bonus information found/bonus challenges
    for k, v in extra_credit.items():
        output += k + ": " + str(v) + "\n"

    output += "\n"
    # Add the list of awards at the bottom
    output += "List of Predicted Awards:\n" + "\n".join(award_list)

    # Write the output to a text file
    with open("human_readable_output.txt", "w") as file:
        file.write(output)

def is_human_name(name):
    if re.search(r'[@#]', name) or name.lower() in {'goldenglobes', 'rt', 'tv', 'movie', 'film'}:
        return False
    return bool(re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)*$", name))

def extract_best_dressed_mentions(texts):
    best_dressed_mentions = []
    for text in texts:
        if 'best dressed' in text.lower():
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'PERSON' and is_human_name(ent.text):
                    best_dressed_mentions.append(ent.text)
    return best_dressed_mentions

def extract_best_joke_mentions(texts):
    joke_mentions = []
    for text in texts:
        if any(phrase in text.lower() for phrase in joke_phrases):
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'PERSON' and is_human_name(ent.text):
                    joke_mentions.append(ent.text)
    return joke_mentions

def get_best_dressed_and_jokes():
    texts = pd.read_csv('text_cleaned.csv')['text'].dropna().tolist()
    best_dressed_mentions = extract_best_dressed_mentions(texts)
    best_dressed_counts = Counter(best_dressed_mentions)
    best_dressed = best_dressed_counts.most_common(1)[0][0] if best_dressed_counts else None

    best_joke_mentions = extract_best_joke_mentions(texts)
    best_joke_counts = Counter(best_joke_mentions)
    best_joke = best_joke_counts.most_common(1)[0][0] if best_joke_counts else None

    return {"best_dressed": best_dressed, "best_joke": best_joke}
