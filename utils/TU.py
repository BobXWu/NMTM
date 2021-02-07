import os
import argparse
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
args = parser.parse_args()

texts = list()
with open(os.path.join(args.data_path)) as file:
    for line in file:
        texts.append(line.strip())

K = len(texts)
T = len(texts[0].split())
vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
counter = vectorizer.fit_transform(texts).toarray()

TU = 0.0
TF = counter.sum(axis=0)
cnt = TF * (counter > 0)

for i in range(K):
    TU += (1 / cnt[i][np.where(cnt[i] > 0)]).sum() / T
TU /= K

print("TU: {:5f}".format(TU))
