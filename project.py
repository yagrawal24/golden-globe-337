import json
import re
from collections import defaultdict
from difflib import SequenceMatcher

def load_json_data(file_path):
    """Load JSON data from a file, ensuring it's read as a dictionary."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            # Convert list to dictionary if the JSON is structured incorrectly
            data = {str(i): item for i, item in enumerate(data)}
        return data

def similarity(a, b):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()

def aggregate_awards(award_dict, threshold=0.85):
    """Aggregate awards by consolidating similar award names."""
    award_keys = sorted(award_dict.keys(), key=len, reverse=True)
    consolidated_awards = {}

    for key in award_keys:
        if key not in award_dict:
            continue

        main_award = award_dict[key]
        consolidated_awards[key] = main_award

        for other_key in list(award_dict.keys()):
            if other_key == key or other_key not in award_dict:
                continue

            if similarity(key, other_key) >= threshold:
                other_award = award_dict[other_key]
                main_award['nominees'].extend(other_award['nominees'])
                main_award['winner'].extend(other_award['winner'])
                del award_dict[other_key]

    return consolidated_awards

def filter_entities(entity_list):
    """Filter out unwanted words from the list of nominees or winners."""
    filtered_entities = [
        entity for entity in entity_list
        if isinstance(entity, str) and entity.istitle() and len(entity) > 1
    ]
    return list(set(filtered_entities))

def merge_nominees_and_winners(nominees_data, winners_data):
    """Merge nominees and winners and then aggregate similar awards."""
    all_awards = defaultdict(lambda: {'nominees': [], 'winner': []})

    # Combine nominees into the all_awards structure
    for award_name, data in nominees_data.items():
        all_awards[award_name]['nominees'].extend(data.get('nominees', []))

    # Combine winners into the all_awards structure
    for award_name, data in winners_data.items():
        all_awards[award_name]['winner'].extend(data.get('winner', []))

    # Aggregate similar awards
    aggregated_awards = aggregate_awards(all_awards)

    # Format the awards for final output
    formatted_awards = {}
    for award_name, data in aggregated_awards.items():
        # Filter and flatten lists for both nominees and winners
        nominees = filter_entities(data['nominees'])
        winners = filter_entities(data['winner'])

        # Select the most frequent winner, if available
        top_winner = max(set(winners), key=winners.count) if winners else None

        formatted_awards[award_name] = {
            "nominees": nominees,
            "winner": top_winner
        }

    return formatted_awards

# Paths to the nominee and winner JSON files
nominees_path = './aggregated_nominees.json'
winners_path = './consolidated_awards.json'

# Load nominee and winner data from JSON files
nominees_data = load_json_data(nominees_path)
winners_data = load_json_data(winners_path)

# Merge nominees and winners, then aggregate
final_awards = merge_nominees_and_winners(nominees_data, winners_data)

# Save the final formatted awards to a JSON file
with open('formatted_awards.json', 'w') as f:
    json.dump(final_awards, f, indent=2)
print("Aggregation complete. Output saved to formatted_awards.json.")
