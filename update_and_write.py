import pandas as pd
import numpy as np 
#import os
from Clean_function import clean_note

from collections import OrderedDict
from progressbar import Percentage, ProgressBar,Bar,ETA

from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
maxlen=2500
min_word_frequency=5
from tensorflow import keras
#Here we will put the text and label Token "updater"

import pandas as pd
from nltk.tokenize import word_tokenize
from progressbar import Percentage, ProgressBar,Bar,ETA
from collections import OrderedDict

#Writing to TF records

#We write the inputs and labels! as integer sequences has a BUNCH of advantges to writing it as multi hot encoded. 
from progressbar import Percentage, ProgressBar,Bar,ETA
import tensorflow as tf 
import pickle
import os.path

def write_tf_records(writer,text_list,label_array,maxlen):
    '''
        Args:
        1. writer: a tf data "writer object" we will use this to "write" the tfrecords on HD
        2. text_list: A list of Medical Notes integer coded 
        3. label_array: A List of Integer coded ICD 9 codes
        6. maxlen the maximum lenght of a note we want to deal with. Longer notes are truncated

        '''
    pbar=ProgressBar(widgets=[Bar('=', '[', ']'), ' ', Percentage(), ' ', ETA()],
                   maxval=len(text_list)).start()
    
    for j in range(0,len(text_list)):
        review=text_list[j]
        target=label_array[j]

        ex = tf.train.SequenceExample()
        label_indexes = ex.feature_lists.feature_list['label_indexes']
        for token_index in target:
            label_indexes.feature.add().int64_list.value.append(token_index)
        token_indexes = ex.feature_lists.feature_list['token_indexes']
        for token_index in review:
            token_indexes.feature.add().int64_list.value.append(token_index)
        writer.write(ex.SerializeToString())
        pbar.update(j)
        
def initial_tokenizer_text(ICD_file_en='/data/ICD9/icd9.txt'
                      ,tokenizer_name='/tokenizer/Text_Tokenizer.pkl'
                      ,min_word_frequency=5):
    #We set up an initial TOkenizer on the Text Descriptions(Italian and english )

    #Get English Descriptions 
    all_names=pd.read_table(ICD_file_en,encoding = "ISO-8859-1")
    ICD9_des_en=all_names['LONG DESCRIPTION'].tolist()+all_names['SHORT DESCRIPTION'].tolist()
    ICD9_des_en=[clean_note(tx) for tx in ICD9_des_en]
    init_text=ICD9_des_en

    print('Initial Set Up  Tokenizer')
    extra_words=[word_tokenize(review) for review in init_text]
    extra_words = [word for review in extra_words for word in review]
    extra_words=list(set(extra_words))
    vocabulary = extra_words

    #i + 1 because 0 wil represent the padding!
    word2idx=OrderedDict(((x,i+1) for i,x in enumerate(vocabulary)))
    t=keras.preprocessing.text.Tokenizer(oov_token='UNK__TO_123')
    t.word_index = word2idx
    t.word_index['UNK__TO_123']=len(vocabulary)+2
    # Write word2index (tokenizer) to disk
    pickle.dump(t, open(tokenizer_name, 'wb'))

def initial_tokenizer_label(tokenizer_name='/tokenizer/Label_Tokenizer.pkl'):
    #we use the keras tokenizer, it is basically a wrapper for the dictionary with some nice added functionality. 
    t=keras.preprocessing.text.Tokenizer()
    t.word_index={}
    # Finally we save the tokenizer. 
    #we leave 0 empty and start the real tokens from 1 
    pickle.dump(t, open(tokenizer_name, 'wb'))
    
    
def Update_Tokenizer_Text(initial_tokenizer,new_text,min_word_frequency=5):

    vocabulary_old=list(initial_tokenizer.word_index.keys())

    #tokenize the new texts, check for most common words. 
    reviews = [word_tokenize(review) for review in new_text]
    len_reviews = [len(review) for review in reviews]

    # Flatten nested list
    reviews = [word for review in reviews for word in review]

    # Compute the frequency of each word
    word_frequency = pd.value_counts(reviews)

    # Keep only words with frequency higher than minimum
    potential_vocabulary = word_frequency[word_frequency>=min_word_frequency].index.tolist()
    #we add words to the vocabulary, that appear at least min_word_frequency in the new docs and which we dont 
    #already have. 
    vocab_add=[item for item in potential_vocabulary if item not in vocabulary_old]
    print('Words added:')
    print(len(vocab_add))
    
    #only if there is something to add we add. 
    if len(vocab_add)>0:
        #then we add them, while perserving order
        vocab_new=vocabulary_old+vocab_add
        
        word2idx_new=OrderedDict(((x,i+1) for i,x in enumerate(vocab_new)))
        initial_tokenizer.word_index = word2idx_new
    return initial_tokenizer

def Update_Tokenizer_Labels(initial_tokenizer,labels):
    
    if len(initial_tokenizer.word_index)>0: 
        
        vocabulary_old=list(initial_tokenizer.word_index.keys())    
        potential_vocabulary = [word for review in labels for word in review]
        #remove duplicates
        potential_vocabulary=list(set(potential_vocabulary))
        #only add labels we havent seen yet
        vocab_add=[item for item in potential_vocabulary if item not in vocabulary_old]
        vocab_add=[word for word in vocab_add if type(word)==str ]
        print('Labels added:')
        print(len(vocab_add))
        #add it IF THERE IS SOMETHING TO ADD
        if len(vocab_add)>0:
            vocab_new=vocabulary_old+vocab_add
            #use ordered dict to keep original order
            word2idx_new=OrderedDict(((x,i+1) for i,x in enumerate(vocab_new)))
            #modify the word index with the new index
            initial_tokenizer.word_index = word2idx_new
            #return tokenizer with new index. 
        
    else:
        print('Empty dictionary: Initial Tokenizer ? ')

        potential_vocabulary = [word for review in labels for word in review]
        #remove duplicates
        potential_vocabulary=list(set(potential_vocabulary))
        vocab_add=[word for word in potential_vocabulary if type(word)==str ]
        vocab_new=vocab_add
        print('Labels added:')
        print(len(vocab_add))
        #use ordered dict to keep original order
        word2idx_new=OrderedDict(((x,i+1) for i,x in enumerate(vocab_new)))
        #modify the word index with the new index
        initial_tokenizer.word_index = word2idx_new
        #return tokenizer with new index. 
    return initial_tokenizer




def update_write(tokenizer_text_path,tokenizer_label_path,In_Text,labels,output_name):
    #First we check for existance of tf records file if yes we break and require a new name
    if os.path.isfile(output_name+'_train.tfrecords')==True:
        return print("File already exists please choose different name.")
        
    
    print('Cleaning Notes')
    In_Text=[clean_note(tx) for tx in In_Text]
    
    print('Update Text Tokenizer')
    initial_tokenizer = pickle.load(open(tokenizer_text_path, 'rb'))
    new_tokenizer=Update_Tokenizer_Text(initial_tokenizer=initial_tokenizer,min_word_frequency=min_word_frequency,new_text=In_Text)
    pickle.dump(new_tokenizer, open(tokenizer_text_path, 'wb'))

    #initial Label tokenizer
    print('Update Label Tokenizer')
    initial_tokenizer_label = pickle.load(open(tokenizer_label_path, 'rb'))
    
    new_tokenizer_label=Update_Tokenizer_Labels(initial_tokenizer_label,labels)
    #Then tokenization, words, labels. 
    pickle.dump(new_tokenizer_label, open(tokenizer_label_path, 'wb'))
    #initial Tokenizer might be empty, we include the handling here. 
    
    
    print('Encode the Data') 
    In_Text,labels=new_tokenizer.texts_to_sequences(In_Text),new_tokenizer_label.texts_to_sequences(labels)
    
    #Shorten texts to max lengths. ( they are already tokenized, dont do this on the strings:D )
    In_Text=[x[-maxlen:] for x in In_Text]
    
    #splitting into train test etc. 
    train_text,val_text,train_labels,val_labels = train_test_split(In_Text, labels, test_size=0.1, random_state=1)
    train_text,test_text,train_labels,test_labels = train_test_split(train_text, train_labels, test_size=0.1, random_state=1)

    
    #define the 3 writers
    writer_train = tf.python_io.TFRecordWriter(  output_name+ '_train.tfrecords')
    writer_validation = tf.python_io.TFRecordWriter( output_name+ '_validation.tfrecords')
    writer_test = tf.python_io.TFRecordWriter( output_name+ '_test.tfrecords')

    #Finally write each TF records file dont forget to close it!
    print('Writing Tf Records Train')
    write_tf_records(writer_train,train_text,train_labels,maxlen)
    writer_train.close()
    print('Writing Tf Records Validation')
    write_tf_records(writer_validation,val_text,val_labels,maxlen)
    writer_validation.close()
    print('Writing Tf Records Test')
    write_tf_records(writer_test,test_text,test_labels,maxlen)
    writer_test.close()