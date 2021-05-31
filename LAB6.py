# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 08:00:02 2021

@author: Hyo-J
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('popular')
              
              
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
import os
os.chdir('E:/SEOULTECH/ML/LAB/')

# Split the text into individual words and create a frequency table and plot. 다음 글을 단어로 쪼개서 빈도테이블과 그래프 그리기
text = "Now, I understand that because it's an election season expectations for what we will achieve this year are low But, Mister Speaker, I appreciate the constructive approach that you and other leaders took at the end of last year to pass a budget and make tax cuts permanent for working \
families. So I hope we can work together this year on some bipartisan priorities like criminal justice reform and helping people who are battling prescription drug abuse and heroin abuse. So, who knows, we might surprise the cynics again"
words = word_tokenize(text)
fdist = FreqDist(words)
fdist.most_common(2)
fdist.plot(30)
plt.show()

# Split the text into sentences and tokenize the sentences and count the number of words. Draw the bar plot. 먼저 문장으로 쪼개고 다시 문장을 단어로 쪼개서 문장별 단어의 갯수를 카운트하고 막대그래프 그리기
sent = sent_tokenize(text)
word_num = []
for s in sent:
    word_num.append(len(word_tokenize(s)))
sns.barplot(x=np.arange(1,4), y=word_num)
plt.title('Number of Words by Sentence')
plt.show()

# 링크에 있는 텍스트를 이용해서 불용어처리, 어간추출, 문장부호를 제거한 후 워드클라우드를 그리시오.
import requests
from bs4 import BeautifulSoup
import string
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud

ps = PorterStemmer()
stop_words = stopwords.words('english')

url = 'http://programminghistorian.github.io/ph-submissions/assets/basic-text-processing-in-r/sotu_text/236.txt'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
text = soup.get_text()

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stop_words]
    return text
words = clean_text(text)
fdist = FreqDist(words)
fdist.plot(20)
plt.show()

wc = WordCloud(background_color='white').generate_from_frequencies(fdist)
plt.imshow(wc)
plt.axis('off')
plt.show()