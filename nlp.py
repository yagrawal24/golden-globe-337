import pandas as pd
import json
from collections import defaultdict

# Award class to store information about each award
class Award:
    def __init__(self, name):
        self.name = name
        self.winner = {}
        self.nominees = {}
        self.presenters = {}
        self.votes = 1
    
    def new_person(self, name, role, pcount=1):
        """Add a new person associated with the award."""
        if role == 'winner':
            self.winner[name] = pcount
        elif role == 'nominee':
            self.nominees[name] = pcount
        elif role == 'presenter':
            self.presenters[name] = pcount
        if pcount > 1:
            self.votes += pcount - 1
    
    def person_vote(self, name, role):
        """Increase the vote count for a person in a specific role."""
        if role == 'winner':
            self.winner[name] += 1
        elif role == 'nominee':
            self.nominees[name] += 1
        elif role == 'presenter':
            self.presenters[name] += 1
    
    def remove_person(self, name, role):
        """Remove a person from a role and adjust the vote count."""
        if role == 'winner':
            pcount = self.winner.pop(name)
            self.votes -= pcount
        elif role == 'nominee':
            pcount = self.nominees.pop(name)
            self.votes -= pcount
        elif role == 'presenter':
            pcount = self.presenters.pop(name)
            self.votes -= pcount
        return name, role, pcount
    
    def award_vote(self):
        """Increase vote count for this award."""
        self.votes += 1
    
    def contains(self, name, role):
        """Check if a person is already in a specific role for this award."""
        if role == 'winner':
            return name in self.winner
        elif role == 'nominee':
            return name in self.nominees
        elif role == 'presenter':
            return name in self.presenters
        return False
    
    def output(self):
        """Prepare the award information for JSON output."""
        return {
            self.name: {
                "nominees": [(n, self.nominees[n]) for n in self.nominees],
                "presenters": [(p, self.presenters[p]) for p in self.presenters],
                "winner": [(w, self.winner[w]) for w in self.winner],
                "votes": self.votes
            }
        }

# Dictionary to hold all awards
awards = {}

# Function to extract answers from the data
def extract_answers(text):
    try:
        nominee, curr_award, role = text.split(' | ')
        
        if curr_award not in awards:
            awards[curr_award] = Award(curr_award)
        else:
            awards[curr_award].award_vote()
            
        if not awards[curr_award].contains(nominee, role):
            awards[curr_award].new_person(nominee, role)
        else:
            awards[curr_award].person_vote(nominee, role)
    
    except ValueError:
        print(f"Skipping invalid entry: {text}")  # Log any problematic entries

# Function to move data from one award to another to prevent duplicates
def move_data(a1: Award, a2: Award):
    """Move people from one award to another to consolidate data."""
    for role, attr in [("winner", a1.winner), ("nominee", a1.nominees), ("presenter", a1.presenters)]:
        for name in list(attr):
            if a2.contains(name, role):
                name, role, pcount = a1.remove_person(name, role)
                a2.new_person(name, role, pcount)

# Load the data and apply the function to extract answers
data = pd.read_csv('award_names_2.csv')
data['Output'].dropna().apply(extract_answers)

# Function to aggregate award data by merging similar award names
def aggregate_awards():
    award_names = list(awards.keys())
    for i in range(len(award_names)):
        for j in range(i + 1, len(award_names)):
            a1 = awards[award_names[i]]
            a2 = awards[award_names[j]]
            # Check if awards are similar enough to be merged
            if a1.name.lower() in a2.name.lower() or a2.name.lower() in a1.name.lower():
                move_data(a1, a2)
                a2.votes += a1.votes
                del awards[a1.name]
                break

# Aggregate awards to remove duplicates
aggregate_awards()

# Convert the final output to JSON
final_output = {award: awards[award].output() for award in awards}

# Save the output to JSON file
with open('unique_award_winners.json', 'w') as f:
    json.dump(final_output, f, indent=4)

print("Unique, non-repetitive award winners saved to 'unique_award_winners.json'.")
