'Full BiLSTM neural network analysis for Twitter data based on word2vec vector space coordinates, written by Teun Cuijpers'

#imports
import numpy as np
import matplotlib.pyplot as plt
import dataset
import gensim
import re
import nltk
#nltk.download('punkt')
import collections
from sklearn.model_selection import train_test_split

#keras imports
from keras.layers import SpatialDropout1D, Dense, Activation, LSTM, Embedding , Dropout
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers.wrappers import Bidirectional

#dataset parameters
OUTPUT_VAR = 'dependentvariable'

#LSTM hyperparametrization
HIDDEN_LAYER_SIZE = 128
BATCH_SIZE = 64
NUM_EPOCHS = 20
RATE_DROP = 0.2

#connect to dataset
db = dataset.connect('sqlite:///DATA_Sample.db')
tables = [db['sample'+ str(nr)] for nr in '1,2,3,4'.split(',')]

#load all tweet texts and outputs into a list
texts, Y = [], []
for table in tables:
    for tweet in table: 
        texts.append(tweet['text'])
        Y.append(int(tweet[OUTPUT_VAR]))
        
#transform Y to categorical var, one-hot encoding format
Y = np_utils.to_categorical(Y)

#perform cleaning, reshape to appropriate format and do exploratory analysis
texts_clean = []
maxlen = 0
num_recs = 0
worddict = {}
counter = collections.Counter()

#regex strings for emoticons
pos_str = r"[:=;X][oO\-]?[D\)\]pP]"
neg_str = r"[:=;X][oO\-]?[\(/\\O]"
    
for t in texts:
    t = re.sub(pos_str,'posemoticon' ,t)
    t = re.sub(neg_str,'negemoticon' ,t)
    t = re.sub(r'&amp;','<en>',t)
    t = re.sub(r'!', '< excl >',t)
    t = re.sub(r'\?', '< ques >',t)
    t = re.sub(r'(?:@[\w_]+)', "<MENTION >", t) 
#    t = re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", "<HASHTAG >", t) 
    t = re.sub(r"http\S+", "<URL >", t) 
    t = re.sub(r'<  >','< >',t)
    t = t.lower()
    t = re.sub(r'[^\w\s]','',t)
    words = nltk.word_tokenize(t)
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:  
        counter[word] += 1
        if word in worddict:
            worddict[word] += 1
        else:
            worddict[word] = 1
    num_recs += 1
    texts_clean.append(words)

print('maximum tweet length:', maxlen)
print('amount unique words in corpus:',len(worddict))
print('total amount tweets in corpus:',num_recs)
VOCAB_SIZE = len(worddict) + 1

#initialize lookup arrays for word to vector, and vector to word
wordtoidx = collections.defaultdict(int)
for idx, word in enumerate(counter.most_common(VOCAB_SIZE)):
    wordtoidx[word[0]] = idx+1 #find the index based on the word you have

idxtoword = {}
for a,b in wordtoidx.items():
    idxtoword[b] = a #reverse the previously created dict

idxtoword[0] = '_unknown_'
VOCAB_SIZE = len(worddict) + 1

#matrix containing all indices
indices = []
for sentence in texts_clean:
    X = [wordtoidx[word] for word in sentence] #build a matrix word word indices for padding
    indices.append(X)

#pad all sentences to max sentence length
indices = sequence.pad_sequences(indices, maxlen = maxlen)

#import word2vec model
model = gensim.models.Word2Vec.load('word2vec.bin')

#fill matrix with embeddings for each word in gensim model
embedding_matrix = model.wv.vectors
VOCAB_SIZE, EMBED_SIZE = embedding_matrix.shape

#split into train and test vars
Xtrain,Xtest,Ytrain,Ytest = train_test_split(indices, Y, test_size=0.2) 

#Build the full model
model = Sequential()
#embedding layer that masks unknown 
model.add(Embedding(VOCAB_SIZE, EMBED_SIZE, input_length = maxlen, mask_zero = True, trainable=False, weights=[embedding_matrix]))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(units = HIDDEN_LAYER_SIZE, dropout = RATE_DROP, recurrent_dropout = RATE_DROP)))
model.add(Dense(3)) #output var is size 3
model.add(Activation('sigmoid'))

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])

results = model.fit(Xtrain,Ytrain, batch_size= BATCH_SIZE, epochs = NUM_EPOCHS, validation_data = (Xtest,Ytest))


plt.plot(results.history['acc'],color = 'g', label = 'Train')
plt.plot(results.history['val_acc'],color = 'b', label = 'Validation')
plt.legend(loc = 'best')
plt.title('Accuracy')
plt.tight_layout()
plt.show()

plt.plot(results.history['loss'],color = 'g', label = 'Train')
plt.plot(results.history['val_loss'],color = 'b', label = 'Validation')
plt.legend(loc = 'best')
plt.title('Loss')
plt.tight_layout()
plt.show()

score, acc = model.evaluate(Xtest,Ytest,batch_size = BATCH_SIZE)
print('Test loss: %.3f, Test accuracy = %.3f'% (score,acc))

#predict 10 random sentences: 
for i in range(10):
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,len(Xtest[idx]))
    ylabel = Ytest[idx]
    ypred = model.predict(xtest)
    sentence = " ".join([idxtoword[x] for x in xtest[0].tolist() if x != 0])
    print(ypred, ylabel, sentence)

