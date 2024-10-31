import pandas as pd
import spacy
import re

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load text_cleaned.csv and award_names.csv
text_cleaned_df = pd.read_csv('./text_cleaned.csv')
award_names_df = pd.read_csv('./award_names.csv')

# Define nominee-related keywords
nominee_keywords = r'\bnominated\b|\bnominee\b|\bnomination\b|\bhope\b|\bwish\b|\bthink\b|\bbelieve\b|\bshould\b'

# Extract award categories
award_categories = award_names_df['Output'].dropna().unique()

print("Loaded award categories and prepared keyword patterns.")

# Function to extract potential nominees based on proximity to awards
def find_nominees_for_award(text, award_categories):
    """Match nominees with awards based on proximity and context in the text."""
    doc = nlp(text)
    potential_nominees = []

    # Check for nominee keywords
    if re.search(nominee_keywords, text, re.IGNORECASE):
        for award in award_categories:
            if award.lower() in text.lower():
                print(f"Found award mention: '{award}' in tweet: {text[:50]}...")  # Progress check
                
                # Identify nominees by looking for relevant entities
                entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PERSON", "WORK_OF_ART", "ORG"]]
                if entities:
                    potential_nominees.extend([ent[0] for ent in entities])

    return potential_nominees if potential_nominees else None

# Apply the nominee extraction to each row in text_cleaned_df
print("Starting nominee extraction from text_cleaned.csv...")
text_cleaned_df['nominees'] = text_cleaned_df['text'].apply(lambda x: find_nominees_for_award(x, award_categories))

# Drop rows without nominees
output_df = text_cleaned_df.dropna(subset=['nominees'])

print("Extraction completed, saving results...")

# Save the output to a new CSV file
output_df[['text', 'nominees']].to_csv('award_nominees.csv', index=False)

print("Saved nominees to 'award_nominees.csv'.")
