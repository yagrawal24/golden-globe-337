#!/usr/bin/env python
# coding: utf-8

# # Our Final Notebook For the Golden GLobes Project - Project 1
# ### Yash Agrawal, Sorie Yillah, Stephen Savas

# In[12]:


# imports

import pandas as pd
import re
import spacy
from collections import Counter
from collections import defaultdict


# We first did data cleaning. This was deleteing data that were not ascii as this would help delete some langauges that had characters that wernt in the english langauge. We also deleted emoji's, links, and excess white space.  

# In[2]:


df = pd.read_json('gg2013.json')['text']

# Define cleaning function
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

df = df.apply(clean)
cleaned_data = df.dropna()
cleaned_data = cleaned_data[cleaned_data.str.strip() != ""]
cleaned_data.to_csv('text_cleaned.csv', index=False)


# Our next goal was to get winners mapped to award names. This meant getting predicted winners to predicted award names and then later mapping these predicted award names to the autograder award names to check our results.
# 
# Our approach was:
# 1) Filter tweets using the keyword "wins"
# 2) Check these filtered tweet's left side for the subject's since the winner is likely to be the subject. Along with that, we aimed to get words that were near the subject so that if it was a long movie name or tv show name, we got a good portion of the name if not all of it. This subject approach also worked with people paired with spacy entity recognition of "People" entity's.

# In[3]:


# Getting the "wins" format data from cleaned data
nlp = spacy.load('en_core_web_sm')
win_keywords = r"(\bwins\b)"

# List of award show names
award_show_names = [
    'GoldenGlobes', 'Golden Globes', 'Oscars', 'Academy Awards', 'Emmys',
    'Grammy Awards', 'BAFTA', 'SAG Awards', 'Tony Awards', 'Cannes Film Festival',
    'MTV Video Music Awards', 'American Music Awards', 'Critics Choice Awards',
    "People's Choice Awards", 'Billboard Music Awards', 'BET Awards',
    'Teen Choice Awards', 'Country Music Association Awards', 'Academy of Country Music Awards',
    'Golden Globe Awards', 'Emmy Awards', 'Grammy', 'Cannes', 'MTV Awards',
]


# In[8]:


file_path = './text_cleaned.csv'
df = pd.read_csv(file_path)
texts = df['text'].dropna().tolist()

# Function to filter names based on a typical human name pattern
def is_human_name(name):
    # Exclude any Twitter-specific handles and common non-human words like "GoldenGlobes"
    if re.search(r'[@#]', name) or name.lower() in {'rt', 'tv', 'movie', 'film'}:
        return False
    # Ensure it looks like a human name (e.g., capitalized first and last name)
    return bool(re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)*$", name))

# Function to extract potential "Best Dressed" mentions from texts
def extract_best_dressed_mentions(texts):
    best_dressed_mentions = []
    for text in texts:
        if 'best dressed' in text.lower():
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'PERSON' and is_human_name(ent.text):
                    best_dressed_mentions.append(ent.text)
    return best_dressed_mentions

# Filter and count the "best dressed" mentions
print("Filtering texts for 'best dressed' mentions...")
filtered_texts = [text for text in texts if 'best dressed' in text.lower()]
print(f"Filtered down to {len(filtered_texts)} texts containing 'best dressed'.")

print("Extracting names from filtered texts...")
best_dressed_mentions = extract_best_dressed_mentions(filtered_texts)
mention_counts = Counter(best_dressed_mentions)

# Find the most frequently mentioned person as "Best Dressed"
if mention_counts:
    most_mentioned = mention_counts.most_common(1)[0]
    print(f"Most mentioned as 'Best Dressed': {most_mentioned[0]} with {most_mentioned[1]} mentions.")
else:
    print("No valid 'Best Dressed' mentions found.")


# In[9]:


# Define keywords and phrases for "Best Joke"
joke_phrases = ["best joke", "funniest joke", "best comedian", "funniest moment"]

# Function to filter names based on a typical human name pattern
def is_human_name(name):
    if re.search(r'[@#]', name) or name.lower() in {'goldenglobes', 'rt', 'tv', 'movie', 'film'}:
        return False
    return bool(re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)*$", name))

# Function to extract mentions of people associated with the "Best Joke"
def extract_best_joke_mentions(texts):
    joke_mentions = []
    for text in texts:
        if any(phrase in text.lower() for phrase in joke_phrases):
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'PERSON' and is_human_name(ent.text):
                    joke_mentions.append(ent.text)
    return joke_mentions

# Filter and count the "Best Joke" mentions
print("Filtering texts for 'Best Joke' mentions...")
filtered_texts = [text for text in texts if any(phrase in text.lower() for phrase in joke_phrases)]
print(f"Filtered down to {len(filtered_texts)} texts containing 'Best Joke' mentions.")

print("Extracting names from filtered texts...")
best_joke_mentions = extract_best_joke_mentions(filtered_texts)
mention_counts = Counter(best_joke_mentions)

# Find the most frequently mentioned person as "Best Joke"
if mention_counts:
    most_mentioned = mention_counts.most_common(1)[0]
    print(f"Most mentioned as 'Best Joke': {most_mentioned[0]} with {most_mentioned[1]} mentions.")
else:
    print("No valid 'Best Joke' mentions found.")


# In[4]:


## Helper functions for extraction of winners/nominees/presenters and award names

# Do this extraction if subj extraction fails
def extract_entities_as_nominee(doc):
    for ent in doc.ents:
        # Consider entities such as PERSON, WORK_OF_ART, ORG, PRODUCT (e.g., "Argo")
        if ent.label_ in ['PERSON', 'WORK_OF_ART', 'ORG', 'PRODUCT']:
            return ent.text
    return None

# Do this extraction of winner first to get subj
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

# Extract the full award name starting with 'Best' using pattern matching and dependency parsing.
# If we see punctuation or VERB we stop capturing since it marks the transition to another sentence part.
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

# Extract the full award name preceding 'award' using pattern matching and dependency parsing.
# If we see punctuation or VERB we stop capturing since it marks the transition to another sentence part.
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

# Extract award name based on two styles: "Best ...." or "... award"
def extract_award_names(text):
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

# Many tweets are RT. Just delete the RT or @ symbol to make parsing and extraction easier.
def ignore_rt_and_mentions(text):
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not (token.text.lower() == 'rt' or token.text.startswith('@'))]
    return ' '.join(filtered_tokens)


# Function to extract winner given a tweet in the format of "X wins Y"
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
        
        if award_category != None and nominee != None:
            return {award_category: nominee}
    
    return None


# In[10]:


def get_winners():
    cleaned_data = pd.read_csv('text_cleaned.csv')['text']
    win_keywords = r"(\bwins\b)"
    win_data = cleaned_data[cleaned_data.apply(lambda x: re.search(win_keywords, x) != None)]
    win_data.to_csv("wins.csv")
    win_output = win_data.apply(find_award_winner)
    win_output = win_output.dropna()
    win_output.to_csv('winners_and_awards.csv')

    return win_output

win_output = get_winners()


# In[11]:


win_output


# In[13]:


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


# In[16]:


def process_win_output(win_output):
    award_data = defaultdict(lambda: {"nominees": [], "presenters": [], "winner": None})
    
    # Iterate through each extracted winner entry to populate award data
    for _, row in win_output.items():  # Use items() for Series iteration
        if isinstance(row, dict):
            for award_name, winner in row.items():
                # Set winner and handle duplicates or additional nominees if necessary
                if not award_data[award_name]["winner"]:
                    award_data[award_name]["winner"] = winner
                else:
                    award_data[award_name]["nominees"].append(winner)
    
    return award_data


# In[17]:


# Apply the process to win_output
structured_award_data = process_win_output(win_output)

# Format the structured data into final output format
formatted_data = format_award_data(structured_award_data)


# In[18]:


structured_award_data


# In[19]:


formatted_data


# (FILL IN INFO FOR NOMINEES)

# Our next goal was to get presenters mapped to award names. This meant getting predicted presenters to predicted award names and then later mapping these predicted award names to the autograder award names to check our results.
# 
# Our approach was:
# 1) Filter tweets using the keywords: "presenter|presenting|presented|presents|present'"
# 2) Similar to wins keywords, we filtern and check for a person entity existing (since presenter will always be a person) and then check for the existence of the word "best" or "award"
# 3) Extract the person and award and store it similar to wins

# In[21]:


# Define presenter-related keywords
presenter_keywords = r'\b(presenter|presenting|presented|presents|present)\b'

# Helper function for extracting presenter entities
def extract_person_entities(text):
    """Extract PERSON entities while excluding Twitter handles and hashtags."""
    doc = nlp(text)
    persons = []
    for ent in doc.ents:
        if ent.label_ == 'PERSON' and not (ent.text.startswith('@') or ent.text.startswith('#')):
            persons.append(ent.text)
    return persons

# Function to infer award names based on common award-related terms
def infer_award_names(text):
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
def extract_presenter_award_pairs(text):
    """Extract presenters associated with inferred awards in a sentence."""
    doc = nlp(text)
    people = extract_person_entities(text)
    award_name = infer_award_names(text)
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

# Main function to process presenter data across cleaned data
def process_presenter_data():
    print("Loading cleaned data...")
    cleaned_df = pd.read_csv('text_cleaned.csv')
    
    # Filter for rows containing presenter keywords
    presenter_data = cleaned_df[cleaned_df['text'].str.contains(presenter_keywords, case=False, na=False)].reset_index(drop=True)
    print(f"Filtered data to {len(presenter_data)} rows with potential presenter mentions.")

    # Extract entities and presenter-award pairs
    print("Extracting presenters and award pairs...")
    presenter_data['Presenters'] = presenter_data['text'].apply(extract_person_entities)
    presenter_data['Presenter_Award_Pairs'] = presenter_data['text'].apply(extract_presenter_award_pairs)
    
    # Filter rows with valid Presenter_Award_Pairs
    presenter_data = presenter_data[presenter_data['Presenter_Award_Pairs'].map(len) > 0]
    print(f"Found {len(presenter_data)} rows with valid presenter-award pairs.")

    # Consolidate presenters per award
    presenter_data['Consolidated_Pairs'] = presenter_data.apply(consolidate_presenters, axis=1)

    # Print consolidated output directly
    for idx, row in presenter_data['Consolidated_Pairs'].items():
        print(row)

    return presenter_data['Consolidated_Pairs']

# Run the presenter processing function
final_output = process_presenter_data()


# In[44]:


## Helper function for presenter extraction 

# Extract PERSON entities from text using spaCy, excluding award show names.
def extract_person_entities(text):
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

# Consolidate presenters per award in the tweet.
def consolidate_presenters(row):
    return row['Presenter_Award_Pairs']

# Extract presenter and award pairs, excluding award show names.
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
    
    # Convert sets to tuples
    award_presenters = {k: tuple(v) for k, v in award_presenters.items()}
    return award_presenters

# Driver function to get presenter-award pairs.
def process_presenter_data():
    cleaned_df = pd.read_csv('text_cleaned.csv')
    presenter_keywords = r'\b(presenter|presenting|presented|presents|present)\b'
    presenter_data = cleaned_df[cleaned_df['text'].str.contains(presenter_keywords, case=False, na=False)]
    presenter_data = presenter_data.reset_index(drop=True)
    
    # Apply entity extraction and pair extraction functions
    presenter_data['Presenters'] = presenter_data['text'].apply(extract_person_entities)
    presenter_data['Presenter_Award_Pairs'] = presenter_data['text'].apply(extract_presenter_award_pairs)
    
    # Keep only rows with non-empty Presenter_Award_Pairs
    presenter_data = presenter_data[presenter_data['Presenter_Award_Pairs'].map(len) > 0]
    
    # Consolidate presenters per award
    presenter_data['Consolidated_Pairs'] = presenter_data.apply(consolidate_presenters, axis=1)
    
    final_output = presenter_data['Consolidated_Pairs']
    final_output.to_csv('presenter_award_consolidated.csv', header=False, index=False)

    return final_output


final_output = process_presenter_data()


# Below is a function to help map the winners and presenters stored in our csv files into the hardcoded dictionary for the autograder.
# It uses Cosine similarity to match our award names and the best similarity predicted award name to the actual award name

# In[50]:


print(final_output)
print(list(final_output))

print(win_output)
print(list(win_output))

for i in win_output:
    print(i.keys())


# In[60]:


# THIS IS ONLY HERE FOR TESTING/AUTOGRADING PURPOSES. THIS HARDCODED LIST WILL BE PASSED BY THE API
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

d2 = {}
for award in award_names:
    d2[award] = None

input_list = win_output

d1 = {}
for i in input_list:
    for k, v in i.items():
        d1[k] = v


import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Step 2: Vectorize the award names using TF-IDF
dict1_keys = list(d1.keys())
dict2_keys = list(d2.keys())
vectorizer = TfidfVectorizer().fit(dict1_keys + dict2_keys)
dict1_vectors = vectorizer.transform(dict1_keys)
dict2_vectors = vectorizer.transform(dict2_keys)

# Step 3: Compute cosine similarity between award names
similarity_matrix = cosine_similarity(dict2_vectors, dict1_vectors)

# Step 4: Find the best match for each award in Dictionary 2
for idx, key2 in enumerate(dict2_keys):
    similarities = similarity_matrix[idx]
    max_sim_idx = similarities.argmax()
    best_match_key = dict1_keys[max_sim_idx]
    d2[key2] = d1[best_match_key]  # Update with winner name

# Step 5: Write the updated Dictionary 2 back to a CSV file
with open('dict2_updated.csv', 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['award_name', 'winner_name']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for award_name, winner_name in d2.items():
        writer.writerow({'award_name': award_name, 'winner_name': winner_name})


# In[ ]:


from rapidfuzz import fuzz

# Initialize dictionary with award names from award_names
d2 = {award: None for award in award_names}
input_list = win_output  # Assuming win_output is a list of dictionaries

# Flatten win_output into d1
d1 = {k: v for item in input_list for k, v in item.items()}

# Use fuzzy string matching to assign the best match
for award_name in d2.keys():
    best_match = None
    best_score = 0
    for key in d1.keys():
        score = fuzz.token_set_ratio(award_name, key)
        if score > best_score:
            best_score = score
            best_match = key
    d2[award_name] = d1[best_match]

# Write the updated Dictionary 2 to a CSV file
with open('dict2_updated.csv', 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['award_name', 'winner_name']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for award_name, winner_name in d2.items():
        writer.writerow({'award_name': award_name, 'winner_name': winner_name})


# In[ ]:




