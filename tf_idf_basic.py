from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re

data=pd.read_csv("music.txt",sep="\t",names=["caption"])

corpus=[]
for i in range(len(data)):
    statement=re.sub("[^a-zA-Z]"," ",data["caption"][i])
    statement=statement.lower()
    statement=statement.split()
    statement=[WordNetLemmatizer().lemmatize(word) for word in statement
               if word not in set(stopwords.words('english'))]
    statement=" ".join(statement)
    corpus.append(statement)

tf_idf=TfidfVectorizer(max_features=5)
mat=tf_idf.fit_transform(corpus)
ary=mat.toarray()
# print(ary)

features=tf_idf.get_feature_names_out()
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
df.to_csv("TF_IDF_Result.csv", index=False)