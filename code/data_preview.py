#!/usr/bin/env python
# coding: utf-8

# # read business and review data

# In[1]:


from os import listdir
from os.path import isfile, join
import json
import csv
import pandas as pd
import numpy as np


# In[3]:


mypath = "../Mod3/data"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles


# In[4]:


business = []
f =  open(mypath+'/'+'business.json', 'r', encoding='utf-8')
for line in f:
    business.append(json.loads(line))
f.close()
business_df = pd.DataFrame(business)


# ## find business: oasis

# In[293]:


business_df.loc[118766, :]['attributes']


# In[8]:


id = business_df.loc[118766, 'business_id']
id


# ## get reviews about The OASIS on Lake Travis

# In[9]:


TOOLT_review = []
f = open(mypath+'/'+'review.json', 'r', encoding='utf-8')
count = 0
for line in f:
    temp = json.loads(line)
    if temp['business_id']==id:
        count += 1
        TOOLT_review.append(temp)
f.close() 


# In[10]:


count


# In[11]:


TOOLT_review_df = pd.DataFrame(TOOLT_review)
TOOLT_review_df.head()


# In[38]:


TOOLT_review_df.to_csv("TOOLT_review.csv")


# ## get tips about The OASIS on Lake Travis

# In[10]:


TOOLT_tips = []
f = open(mypath+'/'+'tip.json', 'r', encoding='utf-8')
count = 0
for line in f:
    temp = json.loads(line)
    if temp['business_id']==id:
        count += 1
        TOOLT_tips.append(temp)
f.close() 


# In[14]:


print(count)
TOOLT_tips_df = pd.DataFrame(TOOLT_tips)
TOOLT_tips_df.head()


# In[15]:


TOOLT_tips_df.to_csv("TOOLT_tip.csv")


# ## get related businesses and their reviews(aren't used in the following code)

# In[12]:


TOOLT = business_df.loc[118766, ]
print(TOOLT.categories)
print(TOOLT.attributes)


# In[13]:


restaurants_list = []
count = 0
for temp in business:
    if temp['categories'] is None:
        continue
    if 'Restaurant' in temp['categories']:
        count += 1
        restaurants_list.append(temp)
print(count)


# In[14]:


Restaurants_and_planing_list = []
count = 0
for temp in business:
    if temp['categories'] is None:
        continue
    if 'Restaurant' in temp['categories'] and 'Plan' in temp['categories']:
        count += 1 
        Restaurants_and_planing_list.append(temp)
print(count)


# In[260]:


Wedding_planing_list = []
count = 0
for temp in business:
    if temp['categories'] is None:
        continue
    if 'Wedding Planning' in temp['categories']:
        count += 1 
        Wedding_planing_list.append(temp)
print(count)


# ## word segregation

# In[40]:


import re 
from PIL import Image
import matplotlib.font_manager as fm
from wordcloud import WordCloud,ImageColorGenerator
from snownlp import SnowNLP
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from nltk.tokenize import word_tokenize
import seaborn as sns
import math
from googletrans import Translator
from nltk.corpus import wordnet as wn


# In[22]:


words = set(nltk.corpus.words.words())
TOOLT_review_df.text = TOOLT_review_df.text.apply(lambda s: " ".join(w.lower() for w in nltk.wordpunct_tokenize(s) if w.lower() in words or not w.isalpha()))


# In[122]:


stemmer = SnowballStemmer("english")
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append(['.',',','us','...','!', 'x x x x','x x x x x',"\'"])
tknzr = TweetTokenizer()#different tokenizer
def get_freq(corpus_review):
    freq = {}
    symbols = r'[0-9!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n]'
    corpus_review=re.sub(symbols, " ", corpus_review)
    words_token = tknzr.tokenize(corpus_review)
    for word in words_token:
        if (len(word) != 1 or word not in stopwords):
#             word = stemmer.stem(word)
            word = word.lower()
            freq[word] = freq.get(word,0) + 1
    return freq


# In[24]:


def get_noun_freq(corpus_review):
    freq = {}
    symbols = r'[0-9!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n]'
    corpus_review=re.sub(symbols, " ", corpus_review)
    words_token = tknzr.tokenize(corpus_review)
    pos_tagged = nltk.pos_tag(words_token)
    nouns = filter(lambda x:x[1]=='NN',pos_tagged)
    nouns_token = map(lambda x: x[0], nouns)
    for word in nouns_token:
        if (len(word) != 1 or word not in stopwords):
#             word = stemmer.stem(word)
            word = word.lower()
            freq[word] = freq.get(word,0) + 1
    return freq


# In[25]:


def freq_matrix(text_list, key_list):
    freq_matrix = {}
    for text, key in zip(text_list, key_list):
        freq_matrix[key] = get_freq(text)
    return freq_matrix


# In[26]:


def nouns_freq_matrix(text_list, key_list):
    freq_matrix = {}
    for text, key in zip(text_list, key_list):
        freq_matrix[key] = get_noun_freq(text)
    return freq_matrix


# In[27]:


def create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = sum(f_table.values())
        for word, count in f_table.items():
#             modified here to get a better result
            tf_table[word] = np.arctan(count / count_words_in_sentence)

        tf_matrix[sent] = tf_table

    return tf_matrix


# In[28]:


def create_idf_matrix(freq_matrix,  total_documents):
    idf_matrix = {}
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(word_per_doc_table[word]))**2

        idf_matrix[sent] = idf_table

    return idf_matrix


# In[29]:


def create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix


# run one of the chunks below

# In[30]:


reviews_by_stars = TOOLT_review_df.groupby('stars')['text'].apply(lambda x: "".join(x)).to_dict()
freq_by_res = freq_matrix(TOOLT_review_df.text, TOOLT_review_df.review_id)
Res_tf = create_tf_matrix(freq_by_res)
Res_idf = create_idf_matrix(freq_by_res, len(TOOLT_review_df))
Res_weight = create_tf_idf_matrix(Res_tf, Res_idf)


# In[275]:



Wedding_planing_list_reviews_df = pd.DataFrame(Wedding_planing_list_reviews)
Wedding_planing_list_reviews_df['text']


# In[134]:


reviews_1=list(TOOLT_review_df.loc[TOOLT_review_df['stars']==1,:]['text'])
reviews_1[0]


# In[135]:


symbols = r'[0-9!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n]'
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append(['.',',','us','...','!', 'x x x x','x x x x x',"\'"])
tknzr = TweetTokenizer()#different tokenizer
for i,x in enumerate(reviews_1):
    x=re.sub(symbols, " ", x)
    words_token = tknzr.tokenize(x)
    words_token=[word for word in words_token if word not in stopwords]
    reviews_1[i]=words_token


# In[139]:



dictionary = corpora.Dictionary(reviews_1)
#print(dictionary.token2id)
#print()
#基于上述字典建立corpus
corpus = [dictionary.doc2bow(text) for text in reviews_1]
print(corpus)# id,num 单词id,出现了几次


# In[155]:


from gensim import models
#tf-idf表达
#初始化tf-idf模型，主要是计算IDF
tfidf = models.TfidfModel(corpus)  
#print(tfidf)

corpus_tfidf = tfidf[corpus]
#for doc, as_text in zip(corpus_tfidf, Comment_cut):
    #print(doc, as_text)
    
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=1)
# 初始化LSI模型参数, K=10

corpus_lsi = lsi_model[corpus_tfidf]
#基于corpus_tfidf训练LSI模型

#打印出学习到的latent topic
#lsi_model.print_topics(3)[0]
lsi_model.print_topics(1)


# In[179]:


def get_topic(score):
    reviews_1=list(TOOLT_review_df.loc[TOOLT_review_df['stars']==score,:]['text'])

    symbols = r'[0-9!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n]'
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append(['.',',','us','...','!', 'x x x x','x x x x x',])
    tknzr = TweetTokenizer()#different tokenizer
    for i,x in enumerate(reviews_1):
        x=re.sub(symbols, " ", x)
        words_token = tknzr.tokenize(x)
        words_token=[word for word in words_token if word not in stopwords]
        words_token=[word for word in words_token if word != "\'"]
        reviews_1[i]=words_token


    dictionary = corpora.Dictionary(reviews_1)
    corpus = [dictionary.doc2bow(text) for text in reviews_1]


    from gensim import models
    tfidf = models.TfidfModel(corpus)  
    corpus_tfidf = tfidf[corpus]
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=1)
    corpus_lsi = lsi_model[corpus_tfidf]
    return lsi_model.print_topics(1)


# In[277]:


Wedding_planing_list_reviews_df.text= Wedding_planing_list_reviews_df.text.apply(lambda s: " ".join(w.lower() for w in nltk.wordpunct_tokenize(s) if w.lower() in words or not w.isalpha()))


# In[280]:


# Food=[food,chicken], Servie=[manager,table,time,order], Place/View
# 


# In[ ]:





# In[294]:



#words = set(nltk.corpus.words.words())
#Wedding_planing_list_reviews_df.text = Wedding_planing_list_reviews_df.text.apply(lambda s: " ".join(w.lower() for w in nltk.wordpunct_tokenize(s) if w.lower() in words or not w.isalpha()))

reviews_1=list(TOOLT_review_df['text'])
k=3
#reviews_1=list(Wedding_planing_list_reviews_df['text'])
symbols = r'[0-9!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n]'
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append(['.',',','us','...','!', 'x x x x','x x x x x'])
tknzr = TweetTokenizer()#different tokenizer
for i,x in enumerate(reviews_1):
    x=re.sub(symbols, " ", x)
    words_token = tknzr.tokenize(x)
    words_token=[word for word in words_token if word not in stopwords]
    words_token=[word for word in words_token if word != "\'" and word !="us"and word !="great" and word !="good" and word !="go"]
    reviews_1[i]=words_token


dictionary = corpora.Dictionary(reviews_1)
corpus = [dictionary.doc2bow(text) for text in reviews_1]


from gensim import models
tfidf = models.TfidfModel(corpus)  
corpus_tfidf = tfidf[corpus]
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=k)
corpus_lsi = lsi_model[corpus_tfidf]
lsi_model.print_topics(k,)


# In[362]:


S=[]
for index, row in TOOLT_review_df.iterrows():
    if "shrimp" in row['text'].lower():
        S.append(row["text"])
S


# In[368]:


I=[]
for s in S:
    try:
        words_token = tknzr.tokenize(s)
        words_token=[word for word in words_token if word not in stopwords]
        i=words_token.index("food")
        #I.append(words_token[i-5:i+5])
        
        tag=nltk.pos_tag(words_token[i-5:i+5])
        re=list(filter(lambda x:x[1]=='JJ',tag))
        re=[x[0] for x in re]
        if re != []:
            I.append(re)
        if any(x in ['unimpressed','bad','poor','disappointed','slow','rude','negative'] for x in re):
            print(s)
            print()
    except:pass


# In[356]:


time=0
for y in I:
    if any(x in ['good','excellent','great','gorgeous','fantasitc','friendly','positive'] for x in y):
        time+=1
print("good",time)
time=0
for y in I:
    if any(x in ['unimpressed','bad','poor','disappointed','slow','rude','negative'] for x in y):
        print(y)
print("bad",time)


# In[298]:


lsi_model.print_topics(1)


# In[291]:


k=5
from gensim import models
tfidf = models.TfidfModel(corpus)  
corpus_tfidf = tfidf[corpus]
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=k)
corpus_lsi = lsi_model[corpus_tfidf]
lsi_model.print_topics(k,)


# In[289]:


mean_star=np.mean(TOOLT_review_df['stars'])
print(mean_star)


# In[363]:


fig, axes = plt.subplots(ncols=2, nrows=2,figsize=(15,8),sharex='row',sharey='row')
W=['cheap',"expensive",'price','time']
for i, ax in zip(range(8), axes.flat):
    
    Score=[]
    for index, row in TOOLT_review_df.iterrows():
        if W[i] in row['text'].lower():
            Score.append(row["stars"])
    print(np.mean(Score))
    value=np.unique(Score,return_counts=True)[0]
    count=np.unique(Score,return_counts=True)[1]   
    count=count/sum(count)        
    ax.bar(value,count)
    ax.set_title(f"{W[i]}")


# In[231]:


from gensim.models import LdaModel
common_dictionary = corpora.Dictionary(reviews_1)
common_corpus = [common_dictionary.doc2bow(text) for text in reviews_1]
# Train the model on the corpus.
lda = LdaModel(common_corpus, num_topics=2)


# In[240]:


dictionary = corpora.Dictionary(reviews_1)
corpus = [dictionary.doc2bow(text) for text in reviews_1]

lda = LdaModel(corpus, id2word=dictionary,num_topics=10)


# In[234]:


lda.get_topics()[0]
[common_dictionary[x[0]] for x in lda.get_topic_terms(0)]


# In[180]:


import re
for score in [1,5]:
    result=get_topic(score)
    print(f"{score} star topic",result)
    print()


# In[157]:


reviews_1=list(TOOLT_review_df.loc[TOOLT_review_df['stars']==5,:]['text'])
reviews_1[0]

symbols = r'[0-9!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n]'
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append(['.',',','us','...','!', 'x x x x','x x x x x',"\'"])
tknzr = TweetTokenizer()#different tokenizer
for i,x in enumerate(reviews_1):
    x=re.sub(symbols, " ", x)
    words_token = tknzr.tokenize(x)
    words_token=[word for word in words_token if word not in stopwords]
    reviews_1[i]=words_token


dictionary = corpora.Dictionary(reviews_1)
#print(dictionary.token2id)
#print()
#基于上述字典建立corpus
corpus = [dictionary.doc2bow(text) for text in reviews_1]
#print(corpus)# id,num 单词id,出现了几次

from gensim import models
#tf-idf表达
#初始化tf-idf模型，主要是计算IDF
tfidf = models.TfidfModel(corpus)  
#print(tfidf)

corpus_tfidf = tfidf[corpus]
#for doc, as_text in zip(corpus_tfidf, Comment_cut):
    #print(doc, as_text)
    
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=1)
# 初始化LSI模型参数, K=10

corpus_lsi = lsi_model[corpus_tfidf]
#基于corpus_tfidf训练LSI模型

#打印出学习到的latent topic
#lsi_model.print_topics(3)[0]
lsi_model.print_topics(1)


# In[34]:


TOOLT_review_df['starofreviews'] = TOOLT_review_df.stars
Res_weight_df = pd.merge(Res_weight_df.reset_index(),
                         TOOLT_review_df[['review_id', 'starofreviews']], 
                         left_on='index', right_on='review_id')


# In[35]:


Res_weight_df = Res_weight_df.drop(['review_id','index'], axis = 1)
Star_Res = Res_weight_df.groupby('starofreviews').sum()


# In[36]:


Star_Res


# In[37]:


def most_common(series):
    print(series.keys()[np.argsort(series.values)][::-1])
    
Star_Res.apply(most_common, axis = 1)


# In[38]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(background_color="white", width=1500, height=960, margin=10)
wordcloud.generate_from_frequencies(frequencies=Star_Res.loc[5].to_dict())
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('wordcloud_5star.png')
plt.show()


# In[39]:


wordcloud = WordCloud(background_color="white", width=1500, height=960, margin=10)
wordcloud.generate_from_frequencies(frequencies=Star_Res.loc[1].to_dict())
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('wordcloud_1star.png')
plt.show()


# In[235]:


VDOT_review_df = pd.DataFrame(VDOT_review)
words = set(nltk.corpus.words.words())
VDOT_review_df.text = VDOT_review_df.text.apply(lambda s: " ".join(w.lower() for w in nltk.wordpunct_tokenize(s) if w.lower() in words or not w.isalpha()))
reviews_by_stars = VDOT_review_df.groupby('stars')['text'].apply(lambda x: "".join(x)).to_dict()
freq_by_res = nouns_freq_matrix(VDOT_review_df.text, VDOT_review_df.review_id)
Res_tf = create_tf_matrix(freq_by_res)
Res_idf = create_idf_matrix(freq_by_res, len(VDOT_review_df))
Res_weight = create_tf_idf_matrix(Res_tf, Res_idf)


# In[237]:


Res_weight_df = pd.DataFrame(Res_weight).T
len(Res_weight_df)


# In[238]:


Res_weight_df.head()


# In[239]:


VDOT_review_df['starofreviews'] = VDOT_review_df.stars
Res_weight_df = pd.merge(Res_weight_df.reset_index(),
                         VDOT_review_df[['review_id', 'starofreviews']], 
                         left_on='index', right_on='review_id')


# In[240]:


Res_weight_df = Res_weight_df.drop(['review_id','index'], axis = 1)
Star_Res = Res_weight_df.groupby('starofreviews').sum()


# In[241]:


Star_Res


# In[242]:


def most_common(series):
    print(series.keys()[np.argsort(series.values)][::-1])
    
Star_Res.apply(most_common, axis = 1)


# In[243]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(background_color="white", width=1500, height=960, margin=10)
wordcloud.generate_from_frequencies(frequencies=Star_Res.loc[5].to_dict())
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('wordcloud_5star.png')
plt.show()


# In[244]:


wordcloud = WordCloud(background_color="white", width=1500, height=960, margin=10)
wordcloud.generate_from_frequencies(frequencies=Star_Res.loc[1].to_dict())
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.savefig('wordcloud_1star.png')
plt.show()


# In[ ]:




