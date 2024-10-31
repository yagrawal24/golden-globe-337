import pandas as pd
import re
from collections import Counter
import spacy

nlp = spacy.load('en_core_web_sm')

cleaned_df = pd.read_csv('text_cleaned.csv')

def extract_retweeted_account(text):
    match = re.match(r'^RT @(\w+):', text)
    return match.group(1) if match else None

cleaned_df['retweeted_account'] = cleaned_df['text'].apply(extract_retweeted_account)

account_rts = cleaned_df['retweeted_account'].value_counts()
top_account = account_rts.head(1).index[0]

host_keywords = r'\b(host|hosts|hosting)\b'

tweets_from_top_account = cleaned_df[cleaned_df['retweeted_account'] == top_account]

host_tweets_top_account = tweets_from_top_account[
    tweets_from_top_account['text'].str.contains(host_keywords, case=False, na=False)
]

def extract_person_names(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

host_tweets_top_account['person_names'] = host_tweets_top_account['text'].apply(extract_person_names)

all_names = [name for names_list in host_tweets_top_account['person_names'] for name in names_list]

name_counts = Counter(all_names)

potential_hosts = name_counts.most_common()

for name, count in potential_hosts:
    print(f"{name}: {count} mentions")
