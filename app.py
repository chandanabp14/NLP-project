import nltk
import spacy
from collections import Counter
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from textblob import TextBlob

nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

article = input("Enter an article: ")
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(article)
print("\nNLTKTokens:",tokens)

stop_words = set(stopwords.words('english'))

filtered_words = []

for word in tokens:
    if word.lower() not in stop_words and word.isalpha():
        filtered_words.append(word.lower())
print("\nFiltered words:",filtered_words)

word_freq = Counter(filtered_words)
print("\nWord Frequency:",word_freq)

doc = nlp(article)
for token in doc:
    print("\nSpacy tokens",token.text)

for token in doc:
    print(token.text, token.pos_)

pos_counts = Counter()
for token in doc:
    pos_counts[token.pos_] += 1
print("\nPOS STATISTICS")
print(pos_counts)

print("\nENTITIES\n")

for ent in doc.ents:
    print(ent.text, ent.label_)

people = []
organizations = []
locations = []

for ent in doc.ents:

    if ent.label_ == "PERSON":
        people.append(ent.text)

    elif ent.label_ == "ORG":
        organizations.append(ent.text)

    elif ent.label_ in ["GPE", "LOC"]:
        locations.append(ent.text)

print("\nPEOPLE")
print(people)

print("\nORGANIZATIONS")
print(organizations)

print("\nLOCATIONS")
print(locations)

analysis = TextBlob(article)

print("\nSentiment Score:")
print(analysis.sentiment)
if analysis.sentiment.polarity > 0:
    print("Positive")

elif analysis.sentiment.polarity < 0:
    print("Negative")

else:
    print("Neutral")