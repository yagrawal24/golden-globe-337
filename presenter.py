import spacy
import re
import pandas as pd

nlp = spacy.load('en_core_web_sm')

# Load the cleaned dataset as a DataFrame (do not select only the 'text' column)
cleaned_df = pd.read_csv('text_cleaned.csv')

# Define presenter-related keywords
presenter_keywords = r'\b(presenter|presenting|presented|presents|present)\b'

# Filter tweets containing presenter-related keywords using the 'text' column
presenter_data = cleaned_df[cleaned_df['text'].str.contains(presenter_keywords, case=False, na=False)]

# Optional: Reset index for easier handling
presenter_data = presenter_data.reset_index(drop=True)

print(f"Number of tweets mentioning presenters: {len(presenter_data)}")

presenter_data.to_csv('presenter_data.csv')

def extract_person_entities(text):
    """Extract PERSON entities from text."""
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    return persons

# Apply the function to the 'text' column in the filtered DataFrame
presenter_data['Presenters'] = presenter_data['text'].apply(extract_person_entities)

# Filter out tweets where no PERSON entities were found
presenter_data = presenter_data[presenter_data['Presenters'].map(len) > 0]

print(f"Number of tweets with person names: {len(presenter_data)}")

import itertools
from collections import Counter

# Flatten the list of presenters
all_presenters = list(itertools.chain.from_iterable(presenter_data['Presenters']))

# Normalize names (strip whitespace and convert to title case)
all_presenters = [name.strip().title() for name in all_presenters]

# Count the frequency of each name
presenter_counts = Counter(all_presenters)

# Convert to DataFrame for better visualization
presenter_counts_df = pd.DataFrame(presenter_counts.items(), columns=['Presenter', 'Count'])

# Sort by count in descending order
presenter_counts_df = presenter_counts_df.sort_values(by='Count', ascending=False).reset_index(drop=True)

presenter_counts_df.to_csv('results.csv')

