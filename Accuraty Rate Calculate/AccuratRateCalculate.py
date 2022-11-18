#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords


# In[38]:


df = pd.read_csv("NLPlabeledData.tsv", delimiter="\t", quoting=3)


# In[5]:


df.head()


# In[6]:


nltk.download('stopwords')


# In[9]:


sample_review= df.review[0]
sample_review


# In[11]:


sample_review = BeautifulSoup(sample_review).get_text()
sample_review


# In[12]:


sample_review = re.sub("[^a-zA-Z]",' ',sample_review)
sample_review


# In[13]:


sample_review = sample_review.lower()
sample_review


# In[14]:


sample_review = sample_review.split()


# In[15]:


sample_review


# In[16]:


len(sample_review)


# In[17]:


swords = set(stopwords.words("english"))
sample_review = [w for w in sample_review if w not in swords]
sample_review


# In[18]:


len(sample_review)


# In[19]:


def process(review):
    # review without HTML tags
    review = BeautifulSoup(review).get_text()
    # review without punctuation and numbers
    review = re.sub("[^a-zA-Z]",' ',review)
    # converting into lowercase and splitting to eliminate stopwords
    review = review.lower()
    review = review.split()
    # review without stopwords
    swords = set(stopwords.words("english"))                      # conversion into set for fast searching
    review = [w for w in review if w not in swords]               
    # splitted paragraph'ları space ile birleştiriyoruz return
    return(" ".join(review))


# In[20]:


train_x_tum = []
for r in range(len(df["review"])):        
    if (r+1)%1000 == 0:        
        print("No of reviews processed =", r+1)
    train_x_tum.append(process(df["review"][r]))


# In[21]:


x = train_x_tum
y = np.array(df["sentiment"])

train_x, test_x, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)


# In[22]:


train_x


# In[23]:


vectorizer = CountVectorizer( max_features = 5000)
train_x = vectorizer.fit_transform(train_x)


# In[24]:


train_x


# In[25]:


train_x = train_x.toarray()


# In[26]:


train_y = y_train
train_x.shape, train_y.shape


# In[27]:


train_y


# In[29]:


model = RandomForestClassifier(n_estimators = 100, random_state=42)
model.fit(train_x, train_y)


# In[30]:


test_xx = vectorizer.transform(test_x)


# In[31]:


test_xx


# In[32]:


test_xx.shape


# In[33]:


test_predict = model.predict(test_xx)
dogruluk = roc_auc_score(y_test, test_predict)


# In[37]:


print("Accuracy rate : :  % ", dogruluk* 100)


# In[ ]:




