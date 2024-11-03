import pandas as pd
import re
import spacy

cleaned_data = pd.read_csv('text_cleaned.csv')['text']

# win_data = cleaned_data[cleaned_data.apply(lambda x: re.search(r'\b(win|wins|won|winner|victor|victory|receive|received|receives|awarded)\b', x) != None)]
win_keywords = r"(\bwins\b)"

win_data = cleaned_data[cleaned_data.apply(lambda x: re.search(win_keywords, x) != None)]

win_data.to_csv("wins.csv")