import numpy as np

from nltk.tokenize import RegexpTokenizer



f=open('English.txt','rb')
text=str(f.read())
tokenizer = RegexpTokenizer(r'\w+')

words=tokenizer.tokenize(text)

f.close()

l_max=30


from collections import Counter

def word2index(words,dictionary):
    #words=text.split()
    if(len(words)>l_max):
        words=words[:l_max]
    wordIndices = []
    for word in words:
        if(word in dictionary):
            wordIndices.append(dictionary[word])
        else:
            wordIndices.append(dictionary['UNK'])
            
    if(len(wordIndices)<l_max):
        wordIndices.extend(np.zeros(l_max-len(wordIndices)))
    return wordIndices	
    

Vocab=Counter(words).most_common(19997)


unzipped = zip(*Vocab)
keys=next(unzipped)
values=range(1,len(keys)+1)
Vocab_dict=dict(zip(keys,values))
Vocab_dict['<START>']=19998
Vocab_dict['<END>']=19999
Vocab_dict['UNK']=20000
vocab_size=len(Vocab_dict.keys())

keys=Vocab_dict.values()
values=Vocab_dict.keys()
reverse_dictionary=dict(zip(keys,values))

X_train=[]
X_train_b=[]
i=0
text=text.split('\\n')

for line in text:
    line=str(line).split()
    line.insert(0,'<START>')
    line.append('<END>')
    p=np.array(word2index(line,Vocab_dict))
    p_rev=p[::-1]
    X_train.append(p)
    X_train_b.append(p_rev)
    i=i+1
		
X_train=np.array(X_train)
X_train_b=np.array(X_train_b)

'''
import pickle
pickle.dump(X_train,open('X_train.pkl','wb'))
pickle.dump(X_train,open('X_train_b.pkl','wb'))

import pickle
X_train=pickle.load(open('X_train.pkl','rb'))
X_train_b=pickle.load(open('X_train_b.pkl','rb'))
'''

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf



num_layers=1
y_max_len=l_max
y_vocab_len=vocab_size

#Encoder Network
num_epochs=50
import keras
import AttentionWithContext as Attention

output_dim=20000
output_length=l_max

model=Attention.AttentionSeq2Seq(output_dim, output_length, batch_input_shape=None,
                                     input_shape=None, input_length=l_max,
                                     input_dim=20000, hidden_dim=200, depth=1,
                                     bidirectional=True, unroll=False, stateful=False, dropout=0.0,)


model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint
ckpt_callback = ModelCheckpoint('keras_model',monitor='val_loss',verbose=1,save_best_only=True,mode='auto')


def process_data(y):
    # Vectorizing each element in each sequence
    return (np.arange(y.max()) == y[...,None]-1).astype(int)  

for k in range(num_epochs):
    with tf.device('/cpu:0'):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        X_train_b = X_train_b[indices]
        y = X_train
        
        for i in range(0, len(X_train), 1000):
            if i + 1000 >= len(X_train):
                i_end = len(X_train)
            else:
                i_end = i + 1000
            xf_sequences=process_data(X_train[i:i_end])
            y_sequences = process_data(y[i:i_end])
            #model.fit([X_train[i:i_end],X_train_b[i:i_end]], y_sequences, batch_size=batch_size, epochs=1, verbose=2,validation_split=0.1,callbacks=[ckpt_callback])
            #model.fit([xf_sequences,xb_sequences,y_sequences], y_sequences, batch_size=100, epochs=2, verbose=2)
            model.fit(xf_sequences, y_sequences, batch_size=100, epochs=1, verbose=2)
            print(xf_sequences[0:1])
            print(model.predict(xf_sequences[0:1]))
            model.save_weights('autoencoder_weights.hdf5')
            