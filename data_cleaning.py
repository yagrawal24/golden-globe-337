import pandas as pd
import re
import emoji
from ftfy import fix_text
import unidecode
# from langdetect import detect, detect_langs

import pandas as pd
import re

import pandas as pd
import re

# Load data
df = pd.read_json('gg2013.json')['text']

# Define cleaning function
def clean(text):
    # Check for foreign language characters (alphabets beyond basic ASCII)
    if re.search(r'[^\x00-\x7F\u263a-\U0001f645]', text):  # Exclude common emoji Unicode ranges
        return None  # Mark as None to drop later
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|pic.twitter\S+', '', text)
    
    # Remove emojis (keep only non-emoji characters)
    text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F]', '', text)
    
    # Remove excess whitespace
    text = re.sub(r' +', ' ', text).strip()
    
    return text

# Apply cleaning function
df = df.apply(clean)

# Remove rows that are None, empty strings, or contain only whitespace
cleaned_data = df.dropna()
cleaned_data = cleaned_data[cleaned_data.str.strip() != ""]

# Save cleaned data to CSV
cleaned_data.to_csv('text_cleaned.csv', index=False)

print("Data cleaned and saved as 'text_cleaned.csv'")

