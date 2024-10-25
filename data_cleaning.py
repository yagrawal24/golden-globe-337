import pandas as pd
import re
import emoji
from ftfy import fix_text
import unidecode
# from langdetect import detect, detect_langs

# df = pd.read_json(r'C:\Users\rockm\golden-globe-337\gg2013.json')['text']
df = pd.read_json('gg2013.json')['text']

df.to_csv('text.csv')

def clean(text):
    text = fix_text(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|pic.twitter\S+', '', text)
    # Remove emojis
    # new_text = unidecode(text)
    text = emoji.replace_emoji(text, replace='')
    # Remove hashtags
    text = re.sub(r'#\S+', '', text)
    
    text = re.sub(r'[^A-Za-z0-9@ "]+', '', text)
    
    text = re.sub(' +', ' ', text)
    
    text = text.lower()
    
    # if text.str.strip() != '':
    #     detect(text.split(' ')[0])
    
    return text

df = df.apply(clean)

cleaned_data = df[df.str.strip() != ""]
cleaned_data.to_csv('text_cleaned.csv')
cleaned_data[cleaned_data.apply(lambda x: re.search('(?=.*award|AWARD)(?=.*wins|Wins|WINS|winner|WINNER).*', x) != None)]