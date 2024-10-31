import json
from collections import defaultdict
from difflib import SequenceMatcher
import pandas as pd

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
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def can_consolidate(name1, name2):
    """Determine if two award names can be consolidated, ensuring Actor/Actress are not merged."""
    keywords = ["Actor", "Actress", "Director", "Screenplay"]
    for keyword in keywords:
        if (keyword in name1 and keyword not in name2) or (keyword in name2 and keyword not in name1):
            return False  # Do not consolidate if keywords are mismatched
    return True  # Otherwise, allow consolidation if similarity threshold is met

def aggregate_awards(award_dict, threshold=0.85):
    """Aggregate awards by consolidating them under the longest, most descriptive award name."""
    # Sort by longest name first for best aggregation
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

            # Enforce consolidation if other_key is a subset or variation of key
            if (other_key in key or similarity(key, other_key) >= threshold) and can_consolidate(key, other_key):
                other_award = award_dict[other_key]
                main_award.consolidate(other_award)
                del award_dict[other_key]  # Remove after consolidation

    return consolidated_awards

def jsonify_output(aggregated_awards):
    """Formats output as JSON with all awards consolidated."""
    final_output = [award.output() for award in aggregated_awards.values()]
    with open('consolidated_awards.json', 'w') as f:
        json.dump(final_output, f, indent=2)
    print("Detailed JSON output:\n", json.dumps(final_output, indent=2))

# Sample function to parse awards data
def parse_award_data(data):
    awards_dict = {}
    for row in data:
        name, award_name, role = row.split(' | ')
        if award_name not in awards_dict:
            awards_dict[award_name] = Award(award_name)
        awards_dict[award_name].add_person(name, role)
    return awards_dict

# Function to generate simplified "NAME | AWARD | ROLE" output and print it
def generate_simple_output(aggregated_awards, role="winner"):
    with open('simplified_awards_output.txt', 'w') as f:
        for award_name, award_obj in aggregated_awards.items():
            for person, count in award_obj.winner.items():  # Assuming 'winner' is the role based on test data
                output_line = f"{person} | {award_name} | {role}"
                print(output_line)  # Print each line to console
                f.write(f"{output_line}\n")

# Function for condensed JSON format
def jsonify_simple_output(aggregated_awards):
    """Generate simplified JSON format with each award's winners, nominees, and presenters."""
    simplified_json_output = {}
    
    for award_name, award_obj in aggregated_awards.items():
        # Create dictionary for each award with structured winners, nominees, and presenters
        simplified_json_output[award_name] = {
            'winner': [person for person, count in award_obj.winner.items()],
            'nominees': [person for person, count in award_obj.nominees.items()],
            'presenters': [person for person, count in award_obj.presenters.items()]
        }
    
    # Print the formatted output to console
    print("\nCondensed JSON format:\n", json.dumps(simplified_json_output, indent=2))

    # Save to JSON file
    with open('formatted_awards_output.json', 'w') as f:
        json.dump(simplified_json_output, f, indent=2)

# Test data
# test_data = [
#     "Christoph Waltz | Best Supporting Actor in a Motion Picture | winner",
#     "Christoph Waltz | Best Supporting Actor | winner",
#     "Christoph Waltz | Best Supporting Actor in Motion Picture | winner",
#     "Daniel Day-Lewis | Best Actor in a Motion Picture | winner",
#     "Jessica Chastain | Best Actress | winner",
#     "Jessica Chastain | Best Actress in a Motion Picture Drama | winner"
# ]

test_data = pd.read_csv("winners_nominees.csv")['text'].dropna().head(200).tolist()

# Parse, aggregate, and save the awards data
awards_dict = parse_award_data(test_data)
aggregated_awards = aggregate_awards(awards_dict)

# 1. Detailed JSON output
jsonify_output(aggregated_awards)

# 2. Simplified "NAME | AWARD | ROLE" output
generate_simple_output(aggregated_awards)

# 3. Condensed JSON format for winners, nominees, and presenters
jsonify_simple_output(aggregated_awards)
