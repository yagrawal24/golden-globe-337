import json
import pandas as pd
import spacy
from collections import defaultdict
from difflib import SequenceMatcher
from capture_group import extract_award_names, extract_entities_as_nominee

# Load SpaCy English language model
print("Loading SpaCy model for nominee extraction...")
nlp = spacy.load('en_core_web_sm')
print("SpaCy model loaded.")

class Award:
    def __init__(self, name):
        self.name = name
        self.nominees = defaultdict(int)
        self.votes = 1

    def add_nominee(self, name, count=1):
        self.nominees[name] += count

    def consolidate(self, other_award):
        for nominee, count in other_award.nominees.items():
            self.nominees[nominee] += count
        self.votes += other_award.votes

    def output(self):
        sorted_nominees = sorted(self.nominees.items(), key=lambda item: item[1], reverse=True)
        return {
            "name": self.name,
            "nominees": [nominee for nominee, _ in sorted_nominees],
            "votes": self.votes
        }

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def can_consolidate(name1, name2):
    keywords = [
        "Actor", "Actress", "Director", "Screenplay", "Singer", "Composer", 
        "Entertainer", "Songwriter", "Performer", "Producer", "Cinematographer", 
        "Editor", "Musician", "Host", "Presenter", "Writer", "Animator", 
        "Designer", "Artist", "Documentary", "Short", "Series", "Feature", 
        "Television", "Film", "Movie", "Drama", "Comedy", "Supporting", 
        "Lead", "Ensemble", "Vocalist", "Cast", "Voice", "Newcomer", 
        "Debut", "Breakthrough", "Soundtrack", "Score", "Original", 
        "Visual Effects", "Stunt", "Choreographer", "Makeup", "Costume", 
        "Production", "Set", "Lighting", "Special Effects", "Casting", 
        "Narrator", "Voiceover", "Reality", "Variety", "Talk Show", "Game Show"
    ]
    for keyword in keywords:
        if (keyword in name1 and keyword not in name2) or (keyword in name2 and keyword not in name1):
            return False
    return True

def aggregate_awards(award_dict, threshold=0.85):
    """Aggregate awards by consolidating similar award names."""
    award_keys = sorted(award_dict.keys(), key=lambda k: (len(k), award_dict[k].votes), reverse=True)
    consolidated_awards = {}

    for key in award_keys:
        if key not in award_dict:
            continue

        main_award = award_dict[key]
        consolidated_awards[key] = main_award

        for other_key in list(award_dict.keys()):
            if other_key == key or other_key not in award_dict:
                continue

            if (other_key in key or similarity(key, other_key) >= threshold) and can_consolidate(key, other_key):
                other_award = award_dict[other_key]
                main_award.consolidate(other_award)
                del award_dict[other_key]

    print("Awards consolidated.")
    return consolidated_awards

def build_known_awards(data):
    """Extract potential award names from the dataset using NLP methods."""
    known_awards = set()
    for text in data:
        award_name = extract_award_names(text)
        if award_name:
            known_awards.add(award_name)
    print("Known awards extracted from data.")
    return list(known_awards)

def extract_nominee_info(tweet, known_awards):
    """Identify nominees and associate them with awards."""
    doc = nlp(tweet)
    nominee = extract_entities_as_nominee(doc)
    award_name = None

    for award in known_awards:
        if award.lower() in tweet.lower():
            award_name = award
            break

    return nominee, award_name

# Load data from CSV file
file_path = './winners_nominees.csv'
print("Loading data from CSV...")
df = pd.read_csv(file_path)
print("Data loaded successfully.")

tweets = df['text'].dropna().tolist()
print("Data preprocessed for award extraction.")

# Build known awards dynamically
known_awards = build_known_awards(tweets)

# Extract nominee information and aggregate data
nominees_dict = {}
for tweet in tweets:
    nominee, award_name = extract_nominee_info(tweet, known_awards)
    if nominee and award_name:
        if award_name not in nominees_dict:
            nominees_dict[award_name] = Award(award_name)
        nominees_dict[award_name].add_nominee(nominee)

# Aggregate similar awards
aggregated_nominees = aggregate_awards(nominees_dict)

# Save results to JSON and text files
print("Saving aggregated nominee data...")
with open('aggregated_nominees.json', 'w') as json_file:
    json.dump({award_name: award_obj.output() for award_name, award_obj in aggregated_nominees.items()}, json_file, indent=2)

with open('aggregated_nominees.txt', 'w') as txt_file:
    for award_name, award_obj in aggregated_nominees.items():
        txt_file.write(f"{award_name}: {award_obj.output()}\n")

print("Nominee data saved to aggregated_nominees.json and aggregated_nominees.txt.")
