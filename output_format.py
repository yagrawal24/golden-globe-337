import json
import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher

class Award:
    def __init__(self, name):
        self.name = name
        self.winner = defaultdict(int)
        self.nominees = defaultdict(int)
        self.presenters = defaultdict(int)
        self.votes = 1

    def add_person(self, name, role, count=1):
        if role == 'winner':
            self.winner[name] += count
        elif role == 'nominee':
            self.nominees[name] += count
        elif role == 'presenter':
            self.presenters[name] += count
        self.votes += count - 1 if count > 1 else 0

    def consolidate(self, other_award):
        for winner, count in other_award.winner.items():
            self.winner[winner] += count
        for nominee, count in other_award.nominees.items():
            self.nominees[nominee] += count
        for presenter, count in other_award.presenters.items():
            self.presenters[presenter] += count
        self.votes += other_award.votes

    def output(self):
        return {
            "name": self.name,
            "winner": list(self.winner.items()),
            "nominees": list(self.nominees.items()),
            "presenters": list(self.presenters.items()),
            "votes": self.votes
        }

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def can_consolidate(name1, name2):
    keywords = ["Actor", "Actress", "Director", "Screenplay"]
    for keyword in keywords:
        if (keyword in name1 and keyword not in name2) or (keyword in name2 and keyword not in name1):
            return False
    return True  

def aggregate_awards(award_dict, threshold=0.85):
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

    return consolidated_awards

def jsonify_output(aggregated_awards):
    """Generate detailed JSON output."""
    final_output = [award.output() for award in aggregated_awards.values()]
    with open('consolidated_awards.json', 'w') as f:
        json.dump(final_output, f, indent=2)

def generate_simple_output(aggregated_awards, role="winner"):
    """Generate simplified "NAME | AWARD | ROLE" format and save to a text file."""
    with open('simplified_awards_output.txt', 'w') as f:
        for award_name, award_obj in aggregated_awards.items():
            for person, count in award_obj.winner.items():
                f.write(f"{person} | {award_name} | {role}\n")

def jsonify_simple_output(aggregated_awards):
    """Generate condensed JSON output for winners, nominees, and presenters."""
    simplified_json_output = {}
    for award_name, award_obj in aggregated_awards.items():
        simplified_json_output[award_name] = {
            'winner': [person for person, count in award_obj.winner.items()],
            'nominees': [person for person, count in award_obj.nominees.items()],
            'presenters': [person for person, count in award_obj.presenters.items()]
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

aggregated_awards = aggregate_awards(awards_dict)
print("Award data aggregated.")

# Generate and save output formats
jsonify_output(aggregated_awards)  # Save to consolidated_awards.json
generate_simple_output(aggregated_awards)  # Save to simplified_awards_output.txt
jsonify_simple_output(aggregated_awards)  # Save to condensed_awards.json

print("All outputs saved successfully.")
