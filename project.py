import spacy
import pandas as pd
import re
import json

# Load SpaCy model without NER for faster processing
print("Loading SpaCy model without NER...")
nlp = spacy.load('en_core_web_sm', disable=['ner'])
print("SpaCy model loaded.")

# Load the cleaned text data
file_path = './text_cleaned.csv'
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)
texts = df['text'].dropna().tolist()
print("Data loaded successfully.")

# Award lexicon components
award_descriptors = [
    "Best", "Outstanding", "Favorite", "Top", "Excellence in", "Achievement in"
]
general_award_categories = [
    "Actor", "Actress", "Director", "Picture", "Screenplay", "Soundtrack",
    "Album", "Song", "Artist", "Performance", "Music Video", "Television Series",
    "Drama", "Comedy", "Animated", "Documentary", "Feature Film", "Reality Show",
    "Supporting Actor", "Supporting Actress"
]
action_keywords = ["win", "won", "nominated", "awarded", "presented"]

# Match award pattern
def match_award_pattern(doc):
    for token in doc:
        if token.lemma_ in action_keywords:
            award_phrase = extract_award_category(doc)
            nominee = extract_nominee_name(doc)
            if award_phrase and nominee:
                return {"nominee": nominee, "category": award_phrase}
    return None

# Extract award category
def extract_award_category(doc):
    for token in doc:
        if token.text in award_descriptors:
            phrase = " ".join([t.text for t in token.subtree])
            for category in general_award_categories:
                if re.search(rf"\b{category}\b", phrase, re.IGNORECASE):
                    return phrase
    return None

# Extract nominee name (fast method without NER)
def extract_nominee_name(doc):
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == "PROPN":  # Proper nouns likely to be nominee names
            return chunk.text
    return None

# Process in batches for efficiency
print("Starting nominee extraction process...")
batch_size = 100  # Adjust batch size based on memory and processing speed
results = []
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_docs = list(nlp.pipe(batch_texts))
    for doc in batch_docs:
        match = match_award_pattern(doc)
        if match:
            results.append(match)

# Save results to JSON file
output_path = 'nominee_extraction_results.json'
print(f"Saving results to {output_path}...")
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("Nominee extraction complete. Results saved to 'nominee_extraction_results.json'")
