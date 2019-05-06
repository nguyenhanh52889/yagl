import pymorphy2
morph = pymorphy2.MorphAnalyzer()

def split_text(text):
    return [x.strip('"({[]})[&?<>/\:;,*^!@#+-=._%$0123457689') for x in text.split()]

def to_normal_form(text):
    result = []
    tags = []
    for word in text:
        parsed = morph.parse(word)[0]
        tag = parsed.tag
        res = parsed.normal_form
        result.append(res)
        tags.append(tag)
    return [result, tags]


import pandas as pd
def change(result, tags):
    X_oh = pd.DataFrame(columns=['keyness_F', 'keyness_M', 'N_F', 'N_M', 'V_F', 'V_M', 'A_F', 'A_M', 'P_F', 'P_M', 'M_F', 'M_M', 'Y_F', 'Y_M', 'YA'])
    X1 = result
    X2 = tags

    F = 0.
    M = 0.

    N_F = 0.
    N_M = 0.

    V_F = 0.
    V_M = 0.

    A_F = 0.
    A_M = 0.

    P_F = 0.
    P_M = 0.

    M_F = 0.
    M_M = 0.

    Y_F = 0.
    Y_M = 0.

    has = 0.
    for word, form in zip(result, tags):
        if word == 'я':
            has = 1.
        if keyword_F.get(word) != None:
            a = keyword_F[word]
        else:
            a = 0.
        F = F + a
        if keyword_M.get(word) != None:
            b = keyword_M[word]
        else:
            b = 0.
        M = M + b


        if "NOUN" in form:
            if "masc" in form:
                N_M += 1.
            if "femn" in form:
                N_F += 1.

        if "VERB" in form:
            add = 1.
            if "1per" in form:
                add = 1000.
            if "masc" in form:
                V_M += add
            if "femn" in form:
                V_F += add

        if "ADJF" in form or "ADJS" in form:
            if "masc" in form:
                A_M += 1.
            if "femn" in form:
                A_F += 1.


        if "NPRO" in form:
            if "masc" in form:
                P_M += 1.
            if "femn" in form:
                P_F += 1.

        if "NUMR" in form:
            if "masc" in form:
                M_M += 1.
            if "femn" in form:
                M_F += 1.

    X_oh = X_oh.append({'keyness_F': F, 'keyness_M': M, 'N_F': N_F, 'N_M':N_M, 'V_F':V_F, 'V_M':V_M, 'A_F':A_F, 'A_M':A_M, 'P_F':P_F, 'P_M':P_M, 'M_F':M_F, 'M_M':M_M, 'Y_F':Y_F, 'Y_M':Y_M, 'YA':has}, ignore_index=True)
    return X_oh

import sys
text = sys.argv[1]
#print(text)
result = to_normal_form(split_text(text))

txt, tag = result[0], result[1]

import json
keyword_F = {}
keyword_M = {}
with open('./keyword.json', 'r') as f:
    dict_vk = json.load(f)
    keyword_F = dict_vk["keyword_F"]
    keyword_M = dict_vk["keyword_M"]

X_oh = change(txt,tag)

import numpy as np
def keyword(keyness, c):
    kw = {}
    for key in keyness.keys():
        if keyness[key] > c:
            kw.update({key: keyness[key]})
    return kw

keyword_F_new = keyword(keyword_F, 1.02)
keyword_M_new = keyword(keyword_M, 1.02)

pos = 0
keywords = {}
for key in keyword_F_new.keys():
    key = key.strip('"({[]})[&?<>/\:;,*^!@#+-=._%$0123457689')
    if keywords.get(key) == None and key[:4] != "http":
        keywords.update({key : pos})
        pos = pos + 1
    
for key in keyword_M_new.keys():
    key = key.strip('"({[]})[&?<>/\:;,*^!@#+-=._%$0123457689')
    if keywords.get(key) == None and key[:4] != "http":
        keywords.update({key : pos})
        pos = pos + 1
kw = [*keywords]

def by_keywords(sentence):
    X_by_kw = np.array([0] * (len(kw)))
    for word in sentence:
        if keywords.get(word) != None:
            X_by_kw[keywords[word]] += 1
    return X_by_kw

X_by_kw = by_keywords(txt)
X_new = np.hstack((X_by_kw, np.array(X_oh.iloc[0].tolist())))

X_new = np.expand_dims(X_new, axis=0)

import pickle
from sklearn.externals import joblib 
#filename = './model.sav'
loaded_model = joblib.load('model.pkl')
result = loaded_model.predict(X_new)
pro = loaded_model.predict_proba(X_new)
print(pro[0][int(result)])
if result == 0:
    print("female")
else:
    print("male")
