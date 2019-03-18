#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys
from keras.models import Model,load_model
from keras.layers import Input, LSTM, Dense
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from nltk.translate.bleu_score import sentence_bleu

# In[2]:


post='stc_weibo_train_post'
response='stc_weibo_train_response'


# In[3]:


batch_size=32
epochs=100
latent_dim=128
num_samples=15000
split_word=" "
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# In[4]:


input_texts=[]
target_texts=[]
input_words=set()
target_words=set()
input_words.add(split_word)
target_words.add(split_word)


# In[6]:


with open(post, encoding = 'utf8') as f:
    post_lines=f.readlines()
with open(response, encoding = 'utf8')  as f:
    response_lines=f.readlines()
for pl,rl in zip(post_lines[:min(num_samples,len(post_lines)-1)],response_lines[:min(num_samples,len(post_lines)-1)]):
    pl=pl.replace("\n","").split(split_word)
    rl=rl.replace("\n","").split(split_word)
    input_texts.append(pl)
    rl=['\t']+rl+['\n']
    target_texts.append(rl)
    for word in pl:
        if word not in input_words:
            input_words.add(word)
    for word in rl:
        if word not in target_words:
            target_words.add(word)
input_words=sorted(list(input_words))
target_words=sorted(list(target_words))
num_encoder_tokens=len(input_words)
num_decoder_tokens=len(target_words)
max_encoder_seq_length=max([len(txt) for txt in input_texts])
max_decoder_seq_length=max([len(txt) for txt in target_texts])
print("Num of samples:",num_samples )
print("Num of unique input token:",num_encoder_tokens)
print("Num of unique ratget token:",num_decoder_tokens)
print("Max seq of input:",max_encoder_seq_length)
print("Max seq of target:",max_decoder_seq_length)


# In[7]:


input_token_index=dict(
    [(word,i) for i,word in enumerate(input_words)]
)
target_token_index=dict(
    [(word,i) for i,word in enumerate(target_words)]
)


# In[8]:


decoder_input_data=np.zeros(
    (len(input_texts),max_decoder_seq_length,num_decoder_tokens),dtype='float32'
)

decoder_target_data=np.zeros(
    (len(input_texts),max_decoder_seq_length,num_decoder_tokens),dtype='float32'
)
encoder_input_data=np.zeros(
    (len(input_texts),max_encoder_seq_length,num_encoder_tokens),dtype='float32'
)


# In[9]:


for i,(input_text,target_text) in enumerate(zip(input_texts,target_texts)):
    for t,word in enumerate(input_text):
        encoder_input_data[i,t,input_token_index[word]]=1.
    for t,word in enumerate(target_text):
        decoder_input_data[i,t,target_token_index[word]]=1.
        if t>0:
            decoder_target_data[i,t-1,target_token_index[word]]=1.


# In[11]:


encoder_inputs=Input(shape=(None,num_encoder_tokens))
encoder=LSTM(latent_dim,return_state=True)
encoder_outputs,state_h,state_c=encoder(encoder_inputs)
encoder_states=[state_h,state_c]


# In[14]:


decoder_inputs=Input(shape=(None,num_decoder_tokens))
decoder_lstm=LSTM(latent_dim,return_sequences=True,return_state=True)
decoder_outputs,_,_=decoder_lstm(decoder_inputs,initial_state=encoder_states)
decoder_dense=Dense(num_decoder_tokens,activation='softmax')
decoder_outputs=decoder_dense(decoder_outputs)


    # In[ ]:

def train():
    model=Model([encoder_inputs,decoder_inputs],decoder_outputs)

    model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
    model.fit([encoder_input_data,decoder_input_data],decoder_target_data,batch_size=batch_size,epochs=epochs,validation_split=0.2)
    model.save('s2s.h5')

if '--train' in sys.argv:
    train()

# In[ ]:

model=load_model('s2s.h5')
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
reverse_input_word_index=dict((i,char) for char,i in input_token_index.items())
reverse_target_word_index=dict((i,char) for char,i in target_token_index.items())


# In[ ]:


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    decoded_sequence=[]
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_word_index[sampled_token_index]
        decoded_sentence += sampled_char
        decoded_sequence.append(sampled_char)
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence,decoded_sequence

for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence,seq = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
    print('BLEU:',sentence_bleu(target_texts, seq))

