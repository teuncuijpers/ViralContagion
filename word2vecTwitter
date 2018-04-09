intro = 'Script for building a word2vec model from microblogging texts, by Teun Cuijpers'
print(intro)

import dataset
from bs4 import BeautifulSoup as bs
import nltk
import re
import gensim
import logging

#connect to db
db = dataset.connect("sqlite:///Data.db")
table = db['Table1']

#load data from db
texts = []
for tweet in table: 
    texts.append(tweet['text'])

maxlen = 0
num_words = 0
worddict = {}
texts_clean = []

#regex for recognizing emoticons:
pos_str = r"[:=;X][oO\-]?[D\)\]pP]"
neg_str = r"[:=;X][oO\-]?[\(/\\O]"

for counter, t in enumerate(texts): 
    #so we know where we are: 
    print(counter/len(table)*100, '%')
    
    #clean text from reading signs using regex
    t = re.sub(pos_str,'posemoticon' ,t)
    t = re.sub(neg_str,'negemoticon' ,t)
    t = re.sub(r'&amp;','<en>',t)
    t = re.sub(r'!', '< excl >',t)
    t = re.sub(r'\?', '< ques >',t)

    t = re.sub(r"http\S+", "<URL >", t) 
    t = re.sub(r'<  >','< >',t)
    t = t.lower()
    t = re.sub(r'[^\w\s]','',t)

    #tokenize texts and make a dictionary with word frequencies
    words = nltk.word_tokenize(t)
    if len(words) > maxlen:
        maxlen = len(words)
    Words = []
    for word in words: 
        Words.append(word)
        if word in worddict:
            worddict[word] += 1
        else:
            worddict[word] = 1
    num_words += len(Words)
    texts_clean.append(Words)
    
    #save cleaned text to db, joined with - :
    table.update(dict(id = Id, text_cleaned = '-'.join(Words)), ['id'])
    

print('Largest text:', maxlen)
print('Vocabulary length:', len(worddict))
print('Total amount of words:', num_words)

#now for the word2vec model: 
#Progress monitor  
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#actual model; only words occurring at least 30 times, 300 vector space
model = gensim.models.Word2Vec(texts_clean, min_count=30, size=300, iter=20)

#inspect model
print('total amount of words in word2vec:', len(model.wv.vocab.keys()))

print(model.wv.most_similar('foo'))
print(model.wv.similarity('foo','bar'))

#save to bin file
model.save('Data_word2vec.bin')

#model = gensim.models.Word2Vec.load('Data_word2vec.bin')
