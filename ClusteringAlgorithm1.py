print("***************************************BI_GRAM CLUSTTERING*******************************************")
import os
import sys
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
sys.path.append("../")

import os
from collections import Counter
from time import time
import re


import numpy as np
import pandas as pd

from tqdm.notebook import tqdm


import nltk
#nltk.download("stopwords")
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
words = set(nltk.corpus.words.words())
f = lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words)

os.chdir("/home/partha/EnvParthaWin/Scripts_NLP")

# ======> Step 1: Read CSV File using pandas ==========>
df=pd.read_csv("startups-data-without-classification.csv",index_col=False)
df_description=df["Description"]
df_keyword=df["Keywords"]
df["Description"].reset_index(drop=True)
df["Keywords"].reset_index(drop=True) 

data= df.fillna(" ")['Description'].astype(str) +" "+ df.fillna(" ")['Keywords'].astype(str)
data_df=data.to_frame(name="DK")                 #=======>Combined Description and Keywords

cols = data_df.select_dtypes(object).columns
data_df_clean = data_df[cols].applymap(f)


DK_all=' '
for item in data_df_clean["DK"]:
    if isinstance(item, str):
        DK_all=DK_all+' '+item

DK_all_super=DK_all





# ===> Step 2: CLEAN DATA        
DK_all = str(DK_all).lower()  # Lowercase words 
DK_all = " ".join(DK_all.split()) 
DK_all=re.sub(r"[^a-zA-Z0-9 ]", " ", DK_all) # Remove Special Char 
DK_all = re.sub(r"\[(.*?)\]", " ", DK_all)            # Remove [+XYZ chars] in content
DK_all = re.sub(r"\w+…|…", "", DK_all)               # Remove ellipsis (and last word)
DK_all=re.sub(r"([A-z])\-([A-z])", r"\1 \2", DK_all) #Replace dash between words
DK_all = re.sub(r"(?<=\w)-(?=\w)", " ", DK_all)      #Replace dash between words
#DK_all=re.sub(r'[^\w\s]', '', DK_all) 
DK_all=re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', ' ', DK_all)        #============>  regex to remove urls
DK_all = re.sub(',', ' ', DK_all)    # Remove Commas
DK_all = re.sub(r"\s+", " ", DK_all)                 # Remove multiple spaces in content 




# Step 3: Run First Spacy Model, Remove stopwords, Remove digits, Remove Unnecessary Words, Lemmatizer        
nlp = spacy.load("en_core_web_md")
nlp.max_length=3595139
DOC = nlp(DK_all)                                       #NLP TRAINING

def remove(list):
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list]
    return list


#==>(3a) Remove StopWords


#stopword_set1={"vow","ax","smm","ax" ,"ei", "myoho", "ot", "batata", "bi", "sku", "chika", "boo", "ub", "ae" ,"dao", "mejor", "saru", "ajo", "ah", "ess", "bei", "ssi", "nuevo", "fai", "paz", "ent", "ahm", "asf" ,"alt", "pai", "co", "aho", "jg", "cia", "dj", "arti", "uli", "iuris", "miel", "al", "nill", "ex", "arp","eg"}                                                                                                        
dflocation = pd.read_csv('/home/partha/EnvParthaWin/Data_NLP/Locations_S_D_C.csv')   
dflocation = dflocation.dropna() 
dflocationCol=dflocation["Locations"].str.lower()    
dflocation_set=set(dflocationCol)   
extrastopwords_set={"vow","ax","smm","ax" ,"ei", "myoho", "ot", "batata", "bi", "sku", "chika", "boo", "ub", "ae" ,"dao", "mejor", "saru", "ajo", "ah", "ess", "bei", "ssi", "nuevo", "fai", "paz", "ent", "ahm", "asf" ,"alt", "pai", "co", "aho", "jg", "cia", "dj", "arti", "uli", "iuris", "miel", "al", "nill", "ex", "arp","eg"}                        
stopword_set_main= extrastopwords_set.union(dflocation_set)  


#df_location = pd.read_csv('gpe_locClean.csv')   
#df_location = df_location.dropna()  
#df_location_column=df_location["Locations"]  
#df_location_column_set=set(df_location_column)
#stopword_set_main = stopword_set1.union(df_location_column_set) 

nlp.Defaults.stop_words |=stopword_set_main                                       

token_list = []
for token in DOC:
    token_list.append(token.text)

DOC_Cleaned =[]     
for word in token_list:
    lexeme = nlp.vocab[word]
    if lexeme.is_stop == False:
        DOC_Cleaned.append(word)



# (3b) Remove Digit
DOC_Cleaned_digit=remove(DOC_Cleaned)        
DOC_Cleaned1 = (" ").join(DOC_Cleaned_digit) 
# print("***********Removing nonunique words************")
# s = DOC_Cleaned1
# l = s.split()
# k = []
# for i in l:
#      if (s.count(i)>=1 and (i not in k)):
#         k.append(i)    
# DOC_Cleaned2=' '.join(k)


# =====>(3c) Removing Unnecessery words    
#DOC_Cleaned3=' '.join(filter(lambda x: x.lower() not in stopword_set_main,  DOC_Cleaned2.split()))     

DOC_Cleaned3=DOC_Cleaned1

# ===> (3d) Lemmatizer (Spacy)
DOC_C_L = []
DOC_C=nlp(DOC_Cleaned3)                                        # NLP Training
for token in DOC_C:
    DOC_C_L.append(token.lemma_)
DOCUMENT_C_L = ' '.join(map(str,DOC_C_L))
DOCUMENT_C_L = re.sub(r"\s+", " ", DOCUMENT_C_L)               # Remove multiple spaces in content


# Step 4: Generate Glove Vectors
Document= nlp(DOCUMENT_C_L)                                   # NLP TRAINING
words = []
for token in Document:
    words.append(token.text) 



# Step 5: Create Bigrams
def create_bigram(tokens):
    # Using words token generated from spacy to find bigram
    bigrams_ = nltk.bigrams(tokens)
    # Convert generator into list of tuples of bigram 
    return list(bigrams_)

bigrams_list = create_bigram(words)
bigrams = [" ".join(bigram) for bigram in list(bigrams_list)]





########

bigrams_element1=[]
bigrams_element2=[]
for token in bigrams:
    token_split=token.split()
    #print(len(token_split))
    if len(token_split)==2:
        #print("True")
        bigrams_element1.append(token_split[0])
        bigrams_element2.append(token_split[1])       
bigrams_element1_all=" ".join(bigrams_element1)
bigrams_element2_all=" ".join(bigrams_element2)



Document1=nlp(bigrams_element1_all) 
Document2=nlp(bigrams_element2_all) 


glove_vectors1 = [w.vector for w in Document1] 
glove_vectors2 = [w.vector for w in Document2]
   
#glove_vectors_F=glove_vectors1+glove_vectors2

words_bigram1 = []
for token in Document1:
    words_bigram1.append(token.text) 
    
words_bigram2 = []
for token in Document2:
    words_bigram2.append(token.text)     



Word_bigram_list=[]
for ii in range(len(words_bigram1)):
    element_bigram1=words_bigram1[ii]
    element_bigram2=words_bigram2[ii]
    element_bigram=element_bigram1+" "+element_bigram2
    Word_bigram_list.append(element_bigram)


#############
glove_vectors_F=[]
for ii in range(len(glove_vectors1)):
    vector1=glove_vectors1[ii]
    vector2=glove_vectors2[ii]
    vector_F=vector1+vector2
    #print(vector_F)
    glove_vectors_F.append(vector_F)
    
    

glove_vectors_F1=[]
for ii  in range(len(Word_bigram_list)):
    bigramvector=nlp(Word_bigram_list[ii]).vector    
    glove_vectors_F1.append(bigramvector)    


# STEP 5: K MEANS CLUSTTERING
def mbkmeans_clusters(X, k, mb=500, print_silhouette_values=False):
    """Generate clusters.

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches. Defaults to 500.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    print("*******************Clusterubng in Progress**********************")
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    return km, km.labels_

k=1
clustering, cluster_labels = mbkmeans_clusters(X=glove_vectors_F1, k=k, print_silhouette_values=True)
df_clusters =pd.DataFrame({"words":Word_bigram_list ,"cluster label": cluster_labels})
df_clusters = df_clusters.drop_duplicates(subset='words', keep="first") # Remove Duplicates


df_Cluster_arranged=[]
for ii in range(k):
    cluster_label_number=ii
    Indvidual_cluster=df_clusters.loc[df_clusters['cluster label'] == cluster_label_number, 'words']
    Indvidual_cluster_with_label={"words":Indvidual_cluster ,"cluster label": cluster_label_number}
    df_I_C=pd.DataFrame(Indvidual_cluster_with_label)
    df_Cluster_arranged.append(df_I_C)
    

# STEP 6: SAVE THE DATA
os.chdir("/home/partha/EnvParthaWin/Data_NLP")
from pandas import ExcelWriter
import xlsxwriter
writer = ExcelWriter('K1CorrectedBigram_1_Cleaned.xlsx')
ii=0
for item in df_Cluster_arranged:
    ii=ii+1
    #item.to_csv('xxxxxx.csv', index = False)
    print(item)    
    item.to_excel(writer, index = False, sheet_name = 'sheet'+str(ii))
print("Writing data")    
writer.save()




















    