import json
import pandas as pd
import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

class Award:
    def __init__(self, name):
        self.name = name
        self.winner = None  # Store as single value instead of defaultdict
        self.nominees = []  # Use list for nominees
        self.presenters = []  # Use list for presenters
        self.votes = 1

    def add_person(self, name, role, count=1):
        if role == 'winner':
            self.winner = name
        elif role == 'nominee':
            self.nominees.append(name)
        elif role == 'presenter':
            self.presenters.append(name)
        self.votes += count - 1 if count > 1 else 0

    def consolidate(self, other_award):
        if other_award.winner:
            self.winner = other_award.winner
        self.nominees.extend(other_award.nominees)
        self.presenters.extend(other_award.presenters)
        self.votes += other_award.votes

    def output(self):
        return {
            "name": self.name,
            "winner": self.winner,
            "nominees": list(set(self.nominees)),  # Remove duplicates
            "presenters": list(set(self.presenters)),  # Remove duplicates
            "votes": self.votes
        }

def aggregate_awards(award_dict, award_names, threshold=0.85):
    """Aggregate awards using cosine similarity to match similar awards."""
    d2 = {award: None for award in award_names}
    d1 = {k: v for k, v in award_dict.items() if k is not None and v is not None}

    # Vectorize the award names and dictionary keys
    all_keys = [key for key in list(d2.keys()) + list(d1.keys()) if key is not None]
    vectorizer = TfidfVectorizer().fit(all_keys)
    award_vectors = vectorizer.transform(list(d2.keys()))
    d1_vectors = vectorizer.transform(list(d1.keys()))

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(award_vectors, d1_vectors)

    # Assign best match
    best_match_indices = np.argmax(similarity_matrix, axis=1)
    for idx, award in enumerate(d2.keys()):
        best_match_key = list(d1.keys())[best_match_indices[idx]]
        d2[award] = d1[best_match_key]

    # Update award_dict with matched awards
    for award_name, winner in d2.items():
        if award_name in award_dict and winner:
            award_dict[award_name].winner = winner
    return award_dict

def jsonify_output(aggregated_awards):
    """Generate detailed JSON output by converting Award objects to dictionaries."""
    final_output = []
    for award in aggregated_awards.values():
        # Check if the object is of type Award and convert it to a dictionary
        if isinstance(award, Award):
            final_output.append(award.output())
        else:
            print("Warning: Non-Award object detected in aggregated_awards.")
            final_output.append(award)  # Directly append if it's not an Award object

    # Save the converted data to JSON
    with open('consolidated_awards.json', 'w') as f:
        json.dump(final_output, f, indent=2)
    print("JSON output saved to 'consolidated_awards.json'")


def generate_simple_output(aggregated_awards, role="winner"):
    """Generate simplified output format for winners."""
    with open('simplified_awards_output.txt', 'w') as f:
        for award_name, award_obj in aggregated_awards.items():
            if award_obj.winner:
                f.write(f"{award_obj.winner} | {award_name} | {role}\n")

def jsonify_simple_output(aggregated_awards):
    """Generate condensed JSON output for winners, nominees, and presenters."""
    simplified_json_output = {}
    for award_name, award_obj in aggregated_awards.items():
        simplified_json_output[award_name] = {
            'winner': award_obj.winner,
            'nominees': list(set(award_obj.nominees)),
            'presenters': list(set(award_obj.presenters))
        }
    with open('condensed_awards.json', 'w') as f:
        json.dump(simplified_json_output, f, indent=2)

def parse_award_data(data):
    awards_dict = {}
    for row in data:
        if pd.notna(row):
            parts = row.split(' | ')
            if len(parts) == 3:
                name, award_name, role = parts
                if award_name not in awards_dict:
                    awards_dict[award_name] = Award(award_name)
                awards_dict[award_name].add_person(name, role)
    return awards_dict

# Load data from CSV file
file_path = './winners_nominees.csv'
df = pd.read_csv(file_path)
print("Data loaded from CSV.")

test_data = df['text'].dropna().tolist()
print("Data preprocessed.")

# Parse, aggregate, and save the awards data
awards_dict = parse_award_data(test_data)
print("Award data parsed.")

# Cosine similarity-based aggregation with hardcoded award names
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

aggregated_awards = aggregate_awards(awards_dict, award_names)
print("Award data aggregated with cosine similarity.")

# Generate and save output formats
jsonify_output(aggregated_awards)
generate_simple_output(aggregated_awards)
jsonify_simple_output(aggregated_awards)

print("All outputs saved successfully.")
