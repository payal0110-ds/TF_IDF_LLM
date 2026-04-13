from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re

messages=pd.read_csv("SMSSpamCollection",sep="\t",names=["label","message"])

corpus=[]
for i in range(len(messages)):
    review=re.sub("[^a-zA-Z]"," ",messages["message"][i])
    review=review.lower()
    review=review.split()
    review=[WordNetLemmatizer().lemmatize(word) for word in review
            if word not in set(stopwords.words('english'))]
    review=" ".join(review)
    corpus.append(review)

tfidf=TfidfVectorizer(max_features=50,ngram_range=(2,3))
mat=tfidf.fit_transform(corpus)
ary=mat.toarray()
# print(ary)

features=tfidf.get_feature_names_out()
# print(features)

# Get the weight of words
for i in range(len(ary)):
    print(f"Document {i+1}:")
    for word, score in zip(features, ary[i]):
        if score > 0:
            print(f"{word}: {score:.4f}")
    print()

# Convert the resukt into a dataframe and export as csv file

df = pd.DataFrame(ary, columns=features)
print(df)
df.to_csv("TFIDF_Result.csv", index=False)

rows = []

for i in range(len(ary)):
    for word, score in zip(features, ary[i]):
        if score > 0:
            rows.append([i+1, word, score])

df_2= pd.DataFrame(rows, columns=["Document", "Word", "TF-IDF"])
print(df_2)
df_2.to_csv("TFIDF_Nonzero.csv", index=False)
