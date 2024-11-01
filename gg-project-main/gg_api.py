'''Version 0.4'''
import pandas as pd
import re
import spacy
from collections import Counter
from collections import defaultdict
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
import csv
import wikipediaapi

# Initialize spaCy model globally
nlp = spacy.load('en_core_web_sm')

wiki = wikipediaapi.Wikipedia('student_homework')

# List of award show names to exclude
award_show_names = [
    'GoldenGlobes', 'Golden Globes', 'Oscars', 'Academy Awards', 'Emmys',
    'Grammy Awards', 'BAFTA', 'SAG Awards', 'Tony Awards', 'Cannes Film Festival',
    'MTV Video Music Awards', 'American Music Awards', 'Critics Choice Awards',
    "People's Choice Awards", 'Billboard Music Awards', 'BET Awards',
    'Teen Choice Awards', 'Country Music Association Awards', 'Academy of Country Music Awards',
    'Golden Globe Awards', 'Emmy Awards', 'Grammy', 'Cannes', 'MTV Awards',
]

irrelevant_words = [
    'kiss', 'son', 'daughter', 'mom', 'dad', 'present', 'win', 'won', 'could', 
    'either', 'famous', 'hosting', 'someone', 'this', 'that', 'cheek', 'when', 
    'nostalgic', 'seeing', 'my', 'your', 'his', 'her', 'their', 'its', 'you', 
    'he', 'she', 'they', 'we', 'me', 'him', 'us', 'who', 'whom', 'it', 's', 've', 'another'
]

common_verbs = ['win', 'wins', 'won', 'receives', 'present', 'presenting']

def clean(text):
    # Check for foreign language characters (alphabets beyond basic ASCII) not including emoji's since those tweets can be useful
    if re.search(r'[^\x00-\x7F\u263a-\U0001f645]', text): 
        return None

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|pic.twitter\S+', '', text)
    
    # Remove emojis (keep only non-emoji characters)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]', '', text)
    
    # Remove excess whitespace
    text = re.sub(r' +', ' ', text).strip()
    
    return text

def extract_entities_as_nominee(doc):
    for ent in doc.ents:
        # Consider entities such as PERSON, WORK_OF_ART, ORG, PRODUCT (e.g., "Argo")
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
    """Extract the award name starting from 'Best' using pattern matching."""
    award_phrases = []
    for i, token in enumerate(doc):
        if token.text.lower() == 'best':
            award_tokens = [token]
            for j in range(i + 1, len(doc)):
                next_token = doc[j]
                # Stop capturing at punctuation, verbs, or if a likely phrase end is reached
                if next_token.text in ('.', ',', ':', ';', '!', '?', '-', 'RT', '@', '#') or next_token.dep_ == 'punct':
                    break
                if next_token.pos_ in ('VERB', 'AUX') and next_token.dep_ in ('ROOT', 'conj'):
                    break
                if next_token.text.lower() == 'for':
                    break
                award_tokens.append(next_token)
            award_phrase = ' '.join([t.text for t in award_tokens]).strip()
            # Add only phrases of a sensible length that could be awards
            if 3 <= len(award_phrase.split()) <= 10:
                award_phrases.append(award_phrase)
    if award_phrases:
        return max(award_phrases, key=len)
    return None

def extract_award_name_before_award(doc):
    """Extract the award name preceding 'award' using pattern matching."""
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
            # Add only phrases of a sensible length that could be awards
            if 3 <= len(award_phrase.split()) <= 10:
                award_phrases.append(award_phrase)
    if award_phrases:
        return max(award_phrases, key=len)
    return None

# def extract_award_names(text):
#     doc = nlp(text)
#     best_award = extract_award_name_after_best(doc)
#     award_name = extract_award_name_before_award(doc)
#     extracted_award = best_award or award_name
#     if extracted_award:
#         # Normalize award name for comparison
#         award_text = extracted_award.strip().lower()
#         award_show_names_lower = [name.lower() for name in award_show_names]
#         if award_text not in award_show_names_lower:
#             return extracted_award
#     return None

def extract_award_names(text):
    """Extract award names using both 'Best' and 'award' triggers and filter irrelevant results."""
    doc = nlp(text)
    best_award = extract_award_name_after_best(doc)
    award_name = extract_award_name_before_award(doc)
    extracted_award = best_award or award_name

    if extracted_award:
        award_text = extracted_award.strip().lower()
        award_words = award_text.split()
        if (
            len(award_words) < 3 or len(award_words) > 10 or
            any(word in irrelevant_words for word in award_words) or
            any(word in common_verbs for word in award_words) or
            not re.search(r'\b(best|actor|actress|director|performance|picture|series|film|song|score)\b', award_text)  # Typical award pattern
        ):
            return None

    return extracted_award

def ignore_rt_and_mentions(text):
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not (token.text.lower() == 'rt' or token.text.startswith('@'))]
    return ' '.join(filtered_tokens)

def find_award_winner(text):
    """Attempt to extract award information and return a structured output."""
    # Ignore 'rt' and mentions but continue with the rest of the tweet
    filtered_text = ignore_rt_and_mentions(text)
    
    doc = nlp(filtered_text)
    
    # Check if the tweet mentions winning or awards
    win_keywords = r"(\bwins\b|\bwon\b|\breceives\b)"
    if re.search(win_keywords, filtered_text, re.IGNORECASE):
        # Extract the nominee (winner)
        nominee = extract_full_subject_as_nominee(doc)
        if not nominee:
            nominee = extract_entities_as_nominee(doc)

        # Extract the award category
        award_category = extract_award_names(doc)
        
        if award_category != None and nominee != None:
            return {award_category: nominee}
    
    return None

def get_winners():
    cleaned_data = pd.read_csv('text_cleaned.csv')['text']
    win_keywords = r"(\bwins\b|\bwon\b|\breceives\b)"
    win_data = cleaned_data[cleaned_data.apply(lambda x: re.search(win_keywords, x, re.IGNORECASE) != None)]
    win_output = win_data.apply(find_award_winner)
    win_output = win_output.dropna()
    return win_output

# def process_win_output(win_output):
#     award_data = defaultdict(lambda: {"nominees": [], "presenters": [], "winner": None})
    
#     # Iterate through each extracted winner entry to populate award data
#     for _, row in win_output.items():  # Use items() for Series iteration
#         if isinstance(row, dict):
#             for award_name, winner in row.items():
#                 # Set winner and handle duplicates or additional nominees if necessary
#                 if not award_data[award_name]["winner"]:
#                     award_data[award_name]["winner"] = winner
#                 else:
#                     # if winner not in award_data[award_name]["nominees"]:
#                         # award_data[award_name]["nominees"].append(winner)
#                     award_data[award_name]["nominees"].append(winner)
    
#     return award_data

def deduplicate_entries(entries):
    unique_entries = []
    seen = set()

    for entry in entries:
        entry_lower = entry.lower()
        
        # Check against seen names using fuzzy matching
        duplicate_found = False
        for unique_entry in unique_entries:
            if fuzz.token_set_ratio(entry_lower, unique_entry.lower()) > 90:  # Adjust threshold as needed
                duplicate_found = True
                break

        # If no close match, add to unique entries
        if not duplicate_found:
            unique_entries.append(entry)
            seen.add(entry_lower)

    return unique_entries

def clean_award_data(award_data):
    cleaned_data = {}
    for award_name, data in award_data.items():
        # Deduplicate nominees, presenters, and winners
        cleaned_data[award_name] = {
            "nominees": deduplicate_entries(data.get("nominees", [])),
            "presenters": deduplicate_entries(data.get("presenters", [])),
            "winner": data.get("winner")
        }
    return cleaned_data

# Modify the function call to apply this clean-up before saving
def process_win_output(win_output):
    award_data = defaultdict(lambda: {"nominees": [], "presenters": [], "winner": None})
    
    # Populate award data
    for _, row in win_output.items():
        if isinstance(row, dict):
            for award_name, winner in row.items():
                if not award_data[award_name]["winner"]:
                    award_data[award_name]["winner"] = winner
                else:
                    award_data[award_name]["nominees"].append(winner)
    
    # Clean and deduplicate data
    return clean_award_data(award_data)


def format_award_data(award_data):
    """Format the award data to match the final submission format."""
    formatted_data = defaultdict(dict)

    for award_name, data in award_data.items():
        formatted_data[award_name] = {
            "nominees": data.get("nominees", []),
            "presenters": data.get("presenters", []),
            "winner": data.get("winner", None)
        }
    
    return formatted_data

def extract_person_entities(text):
    """Extract PERSON entities while excluding Twitter handles and hashtags."""
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

def extract_presenter_award_pairs(text):
    doc = nlp(text)
    people = extract_person_entities(text)
    award_name = extract_award_names(text)
    
    if not award_name:
        return {}
    
    award_presenters = {}
    
    award_show_names_lower = set(name.lower() for name in award_show_names)
    
    presenter_keywords = {'presenting', 'presented', 'presents', 'present'}
    
    for sent in doc.sents:
        sentence_text = sent.text.lower()
        if any(keyword in sentence_text for keyword in presenter_keywords):
            for person in people:
                person_lower = person.strip().lower()
                if person_lower in award_show_names_lower:
                    continue 
                if person_lower in sentence_text:
                    award_presenters.setdefault(award_name, set()).add(person)
    
    # Convert sets to lists
    award_presenters = {k: list(v) for k, v in award_presenters.items()}
    return award_presenters

def consolidate_presenters(row):
    presenter_award_pairs = row['Presenter_Award_Pairs']
    consolidated = defaultdict(set)

    for award, presenters in presenter_award_pairs.items():
        consolidated[award].update(presenters)

    return {award: list(presenters) for award, presenters in consolidated.items()}

def process_presenter_data():
    cleaned_df = pd.read_csv('text_cleaned.csv')
    presenter_keywords = r'\b(presenter|presenting|presented|presents|present)\b'

    # Extract rows containing presenter-related keywords and match the pattern
    presenter_data = cleaned_df[cleaned_df['text'].str.extract(f'({presenter_keywords})', flags=re.IGNORECASE).notnull().any(axis=1)].copy()
    presenter_data['presenter_matches'] = presenter_data['text'].str.extract(f'({presenter_keywords})', flags=re.IGNORECASE)[0]
    presenter_data['Presenters'] = presenter_data['text'].apply(extract_person_entities)
    presenter_data['Presenter_Award_Pairs'] = presenter_data['text'].apply(extract_presenter_award_pairs)
    
    presenter_data = presenter_data[presenter_data['Presenter_Award_Pairs'].map(len) > 0].copy()
    presenter_data['Consolidated_Pairs'] = presenter_data.apply(consolidate_presenters, axis=1)
    
    final_output = defaultdict(set)
    for pairs in presenter_data['Consolidated_Pairs']:
        for award, presenters in pairs.items():
            final_output[award].update(presenters)
    
    final_output = {award: list(presenters) for award, presenters in final_output.items()}
    return final_output

def lookup_wikipedia(name):
    page = wiki.page(name)
    if page.exists():
        return re.search("born", page.summary, re.IGNORECASE) is not None
    return False

def extract_person_names(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

def get_hosts_list():
    cleaned_data = pd.read_csv('./text_cleaned.csv')['text']
    host_keywords = r'\b(host|hosts|hosting)\b'
    
    # Extract host data matching the keywords without triggering regex warnings
    host_data = cleaned_data[cleaned_data.str.extract(f'({host_keywords})', flags=re.IGNORECASE).notnull().any(axis=1)].copy()
    
    # Process first 10% of the data to capture names mentioned early in the tweets
    early_host_data = host_data[:int(0.1 * len(host_data))].apply(extract_person_names)
    
    # Flatten list of names and count occurrences
    all_names = [name for names_list in early_host_data for name in names_list]
    name_counts = Counter(all_names)
    
    # Determine likely hosts by verifying their Wikipedia pages
    hosts = []
    potential = name_counts.most_common()
    i = 0
    while len(hosts) < 2 and i < len(potential):
        current_name = potential[i][0]
        if lookup_wikipedia(current_name):
            hosts.append(current_name)
        i += 1
    
    return hosts

def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''

    # Load data
    df = pd.read_json('gg2013.json')['text']

    # Clean data
    df = df.apply(clean)
    cleaned_data = df.dropna()
    cleaned_data = cleaned_data[cleaned_data.str.strip() != ""]
    cleaned_data.to_csv('text_cleaned.csv', index=False)

    # Process winners
    win_output = get_winners()

    # Process the win_output
    structured_award_data = process_win_output(win_output)

    # Format the data
    formatted_winner_data = format_award_data(structured_award_data)

    # Store the winners
    with open('winners.json', 'w') as f:
        json.dump(formatted_winner_data, f)

    # Process presenters
    presenter_output = process_presenter_data()

    # Store the presenters
    with open('presenters.json', 'w') as f:
        json.dump(presenter_output, f)

    # Extract hosts
    hosts = get_hosts_list()
    # Store hosts
    with open('hosts.json', 'w') as f:
        json.dump(hosts, f)

    # Extract awards
    award_names_from_winners = list(formatted_winner_data.keys())
    award_names_from_presenters = list(presenter_output.keys())
    all_extracted_award_names = set(award_names_from_winners + award_names_from_presenters)

    with open('awards.json', 'w') as f:
        json.dump(list(all_extracted_award_names), f)

    print("Pre-ceremony processing complete.")
    return

def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''

    print("Getting hosts")

    # Read hosts from 'hosts.json'
    with open('hosts.json', 'r') as f:
        hosts = json.load(f)
    return hosts

def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''

    print("Getting awards")

    with open('awards.json', 'r') as f:
        awards = json.load(f)
    return awards

def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''

    print("Getting nominees")

    # Read the winners from 'winners.json'
    with open('winners.json', 'r') as f:
        formatted_winner_data = json.load(f)
    
    # Extract nominees
    extracted_awards = {award: data['nominees'] for award, data in formatted_winner_data.items()}

    # The hardcoded award names
    award_names = [
        "best screenplay - motion picture",
        "best director - motion picture",
        "best performance by an actress in a television series - comedy or musical",
        "best foreign language film",
        "best performance by an actor in a supporting role in a motion picture",
        "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television",
        "best motion picture - comedy or musical",
        "best performance by an actress in a motion picture - comedy or musical",
        "best mini-series or motion picture made for television",
        "best original score - motion picture",
        "best performance by an actress in a television series - drama",
        "best performance by an actress in a motion picture - drama",
        "cecil b. demille award",
        "best performance by an actor in a motion picture - comedy or musical",
        "best motion picture - drama",
        "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television",
        "best performance by an actress in a supporting role in a motion picture",
        "best television series - drama",
        "best performance by an actor in a mini-series or motion picture made for television",
        "best performance by an actress in a mini-series or motion picture made for television",
        "best animated feature film",
        "best original song - motion picture",
        "best performance by an actor in a motion picture - drama",
        "best television series - comedy or musical",
        "best performance by an actor in a television series - drama",
        "best performance by an actor in a television series - comedy or musical"
    ]

    # Initialize dictionary with award names from award_names
    nominees = {award: [] for award in award_names}

    # Use fuzzy string matching to assign the best match
    for award_name in nominees.keys():
        best_match = None
        best_score = 0
        for key in extracted_awards.keys():
            score = fuzz.token_set_ratio(award_name.lower(), key.lower())
            if score > best_score:
                best_score = score
                best_match = key
        nominees[award_name] = extracted_awards.get(best_match, [])

    return nominees

def get_winner(year):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''

    print("Getting winners")

    # Read the winners from 'winners.json'
    with open('winners.json', 'r') as f:
        formatted_winner_data = json.load(f)
    
    # Extract the winner names
    extracted_awards = {award: data['winner'] for award, data in formatted_winner_data.items()}

    # The hardcoded award names
    award_names = [
        "best screenplay - motion picture",
        "best director - motion picture",
        "best performance by an actress in a television series - comedy or musical",
        "best foreign language film",
        "best performance by an actor in a supporting role in a motion picture",
        "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television",
        "best motion picture - comedy or musical",
        "best performance by an actress in a motion picture - comedy or musical",
        "best mini-series or motion picture made for television",
        "best original score - motion picture",
        "best performance by an actress in a television series - drama",
        "best performance by an actress in a motion picture - drama",
        "cecil b. demille award",
        "best performance by an actor in a motion picture - comedy or musical",
        "best motion picture - drama",
        "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television",
        "best performance by an actress in a supporting role in a motion picture",
        "best television series - drama",
        "best performance by an actor in a mini-series or motion picture made for television",
        "best performance by an actress in a mini-series or motion picture made for television",
        "best animated feature film",
        "best original song - motion picture",
        "best performance by an actor in a motion picture - drama",
        "best television series - comedy or musical",
        "best performance by an actor in a television series - drama",
        "best performance by an actor in a television series - comedy or musical"
    ]

    # Initialize dictionary with award names from award_names
    winners = {award: None for award in award_names}

    # Use fuzzy string matching to assign the best match
    for award_name in winners.keys():
        best_match = None
        best_score = 0
        for key in extracted_awards.keys():
            score = fuzz.token_set_ratio(award_name.lower(), key.lower())
            if score > best_score:
                best_score = score
                best_match = key
        winners[award_name] = extracted_awards.get(best_match)

    return winners

def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''

    print("Getting presenters")

    # Read the presenters from 'presenters.json'
    with open('presenters.json', 'r') as f:
        presenter_output = json.load(f)
    
    # The hardcoded award names
    award_names = [
        "best screenplay - motion picture",
        "best director - motion picture",
        "best performance by an actress in a television series - comedy or musical",
        "best foreign language film",
        "best performance by an actor in a supporting role in a motion picture",
        "best performance by an actress in a supporting role in a series, mini-series or motion picture made for television",
        "best motion picture - comedy or musical",
        "best performance by an actress in a motion picture - comedy or musical",
        "best mini-series or motion picture made for television",
        "best original score - motion picture",
        "best performance by an actress in a television series - drama",
        "best performance by an actress in a motion picture - drama",
        "cecil b. demille award",
        "best performance by an actor in a motion picture - comedy or musical",
        "best motion picture - drama",
        "best performance by an actor in a supporting role in a series, mini-series or motion picture made for television",
        "best performance by an actress in a supporting role in a motion picture",
        "best television series - drama",
        "best performance by an actor in a mini-series or motion picture made for television",
        "best performance by an actress in a mini-series or motion picture made for television",
        "best animated feature film",
        "best original song - motion picture",
        "best performance by an actor in a motion picture - drama",
        "best television series - comedy or musical",
        "best performance by an actor in a television series - drama",
        "best performance by an actor in a television series - comedy or musical"
    ]

    # Initialize dictionary with award names from award_names
    presenters = {award: [] for award in award_names}

    # Use fuzzy string matching to assign the best match
    for award_name in presenters.keys():
        best_match = None
        best_score = 0
        for key in presenter_output.keys():
            score = fuzz.token_set_ratio(award_name.lower(), key.lower())
            if score > best_score:
                best_score = score
                best_match = key
        presenters[award_name] = presenter_output.get(best_match, [])

    return presenters

def main():
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''
    # For demonstration purposes, we can call the functions and print the results
    year = 2013
    # hosts = get_hosts(year)
    hosts = [host.lower() for host in get_hosts(year)]
    awards = get_awards(year)
    nominees = get_nominees(year)
    winners = get_winner(year)
    presenters = get_presenters(year)

    print("Results obtained")

    # Assemble the award data, normalizing and verifying content
    award_data = {}
    for award in awards:
        award_lower = award.lower()  # Normalize each award name
        award_data[award_lower] = {
            "nominees": [nominee.lower() for nominee in nominees.get(award, []) if nominee],
            "presenters": [presenter.lower() for presenter in presenters.get(award, []) if presenter],
            "winner": winners.get(award, "").lower() if winners.get(award) else None
        }

    # Build final output dictionary
    final_output = {
        "hosts": hosts,
        "award_data": award_data
    }

    # Save final consolidated JSON output
    with open('final_output.json', 'w') as f:
        json.dump(final_output, f, indent=2)

    print("Final output saved to final_output.json")

    return

if __name__ == '__main__':
    pre_ceremony()
    main()
