## Here we define the Keras Custom Layers that make up the model. 


import tensorflow as tf

import numpy as np
# Import pandas for data processing and pickle for data reading
import pandas as pd
import pickle
import math
from tensorflow import keras
import logging
import numpy as np 

from tensorflow.python.keras import Model,optimizers,initializers
from tensorflow.python.keras.layers import Input,Embedding,Bidirectional,BatchNormalization,Dense,TimeDistributed,Permute,Activation,Add,Dot,Dropout,Lambda,Conv1D
from tensorflow.python.keras import backend as K

if tf.test.is_gpu_available():
    from tensorflow.python.keras.layers import CuDNNLSTM

from tensorflow.python.keras.layers import LSTM

from tensorflow.python.keras.callbacks import LambdaCallback
from tensorflow.python.keras.callbacks import Callback

#from keras.callbacks import LambdaCallback
#from keras.callbacks import Callback

#from keras.layers import Input,Embedding,Bidirectional,Dense,TimeDistributed,Permute,LSTM,Activation,Add,Dot,Dropout,Lambda
#from keras import 
import h5py
#get the function for evlauation
from sklearn.metrics import roc_auc_score,f1_score,log_loss,precision_score,recall_score
import pickle 

def pad_next(code):
    def nextDiv32(n):
        k = n%3
        if (k==0):
            return n
        if (k==1):
            return n+2
        else:
            return n+1
    return keras.preprocessing.sequence.pad_sequences(code,nextDiv32(len(code[0])))

#The Loop Projection Layer Transforms each Aggregation seperately
class Loop_Projection(keras.layers.Layer):

    def __init__(self,projection_dim=None, **kwargs):
        self.projection_dim = projection_dim
        #self.share_paramerters=share_paramerters
        super(Loop_Projection, self).__init__(**kwargs)

        
    def build(self, input_shape):
        
         # Weights initializer function
        w_initializer = keras.initializers.glorot_uniform()

        # Biases initializer function
        b_initializer = keras.initializers.Zeros()
        
        self.nproto=int(input_shape[2])
        #if self.share_paramerters==True:
        #    np=1
        #else:
        #    
        np=int(input_shape[2])
        for j in range(np):
            setattr(self, 'weigth_'+str(j), self.add_weight(name='feature_extract'+str(j), 
                                      shape=(int(input_shape[1]),int(self.projection_dim)),
                                      initializer=w_initializer,
                                      trainable=True))
            
            setattr(self, 'bias'+str(j),self.add_weight(name='feaure_bias'+str(j), 
                                      shape=(int(1),int(self.projection_dim)),
                                      initializer=b_initializer,
                                      trainable=True))
        
        super(Loop_Projection, self).build(input_shape)  # Be sure to call this somewhere!

        
    def call(self,x):
           

        store_out=[]
        for y in range(self.nproto):
            #if self.share_paramerters==True:
            #    i=0
            #else:
            #    i=y
            store_out.append(tf.tensordot(x[:,:,y],getattr(self, 'weigth_'+str(y)),axes=[1,0])+getattr(self, 'bias'+str(y)))
            #store_out.append(K.dot(x[:,:,y],getattr(self, 'weigth_'+str(j)))

        preds2 = tf.stack(store_out,axis=2)

        return preds2

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.projection_dim,input_shape[2])
        #return (input_shape[0],self.projection_dim)
        
    def get_config(self):
        base_config = super(Loop_Projection, self).get_config()
        base_config['projection_dim'] = self.projection_dim
        return base_config

#the Multi Attention layer Forms multiple Attention aggregations of a feature sequence.    
class Multi_Attention(keras.layers.Layer):

    def __init__(self, feature_dim=None,value_dim=None,nheads=None, **kwargs):
        
        self.value_dim=value_dim
        self.nheads=nheads
        self.feature_dim = feature_dim
        super(Multi_Attention, self).__init__(**kwargs)

        
    def build(self, input_shape):
        
         # Weights initializer function
        w_initializer = keras.initializers.glorot_uniform()

        # Biases initializer function
        b_initializer = keras.initializers.Zeros()
        
        #THE Matrix that will generate a feature tesnor for each timestep.
        self.feature_extract = self.add_weight(name='feature_extract', 
                                      shape=(int(input_shape[2]),int(self.feature_dim)),
                                      initializer=w_initializer,
                                      trainable=True)
        
        self.value_extract = self.add_weight(name='feature_extract', 
                                      shape=(int(input_shape[2]),int(self.value_dim)),
                                      initializer=w_initializer,
                                      trainable=True)
        
        #THE Matric that will get us down to one dimension
        self.multi_Query = self.add_weight(name='multi_Query', 
                                      shape=(int(self.feature_dim),int(self.nheads)),
                                      initializer=w_initializer,
                                      trainable=True)
        
        self.feaure_bias = self.add_weight(name='feaure_bias', 
                                      shape=(int(1),int(self.feature_dim)),
                                      initializer=b_initializer,
                                      trainable=True)
        self.value_bias = self.add_weight(name='feaure_bias', 
                                      shape=(int(1),int(self.value_dim)),
                                      initializer=b_initializer,
                                      trainable=True)
        
        self.scale = self.add_weight(name='feaure_bias', 
                                      shape=(int(1),int(1)),
                                      initializer=keras.initializers.Constant(value=int(np.floor(np.sqrt(self.feature_dim))))
                                     ,trainable=False)
        
        super(Multi_Attention, self).build(input_shape)  # Be sure to call this somewhere!

        
    def call(self, x):
        #first we do a "time distributed" dot prodcutplus bias plus tanh to get 
        #a "feature sequence"
        
        ###########Aux High Level Classification ##############

        feature_rep=tf.nn.tanh(tf.tensordot(x,self.feature_extract,axes=[2,0])+self.feaure_bias)
        
        value=tf.nn.tanh(tf.tensordot(x,self.value_extract,axes=[2,0])+self.value_bias)
        
        
        
        similar_logits=tf.tensordot(feature_rep,self.multi_Query,axes=[2,0])
        #/self.scale
        
        attention_weights = tf.nn.softmax(similar_logits,axis=1)
        
        weighted_input = tf.matmul(value, attention_weights, transpose_a=True)
        #tf.squeeze(
        
        
        return weighted_input,attention_weights

    def compute_output_shape(self, input_shape):
        return [(input_shape[0],self.value_dim,self.nheads),(input_shape[0],input_shape[0],self.nheads)]
    
    
    def get_config(self):
        base_config = super(Multi_Attention, self).get_config()
        base_config['feature_dim'] = self.feature_dim
        base_config['value_dim'] = self.value_dim
        base_config['nheads'] = self.nheads
        return base_config

    
#We wrap the summation in a custom layer just to make sure. Lambda layers might break stuff
class global_bias(keras.layers.Layer):

    def __init__(self, **kwargs):

        
        super(global_bias, self).__init__(**kwargs)

        
    def build(self, input_shape):
       
        b_initializer = keras.initializers.Ones()
        self.bias = self.add_weight(name='bias', 
                                      shape=(int(input_shape[1])),
                                      initializer=b_initializer,
                                      trainable=True)       
        
    def call(self, x):
        return x-4*self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


class sum_dum(keras.layers.Layer):

    def __init__(self, **kwargs):

        
        super(sum_dum, self).__init__(**kwargs)

        
    def build(self, input_shape):
        
        
        super(sum_dum, self).build(input_shape)  # Be sure to call this somewhere!

        
    def call(self, x):
        return tf.reduce_sum(x,axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1])
    
    
#This Learnng rate Scheduler was generally usefull for Better convergence
class LearningRateScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.count=1
    def on_epoch_begin(self, epoch, logs=None):
            self.count= epoch*419/2

    def on_batch_begin(self, batch, logs=None):
    #We just update every 30 batches.
        if batch%2 == 0:
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
                
            lr = self.schedule(self.count)
            self.count=self.count+1
            if not isinstance(lr, (float, np.float32, np.float64)):
                raise ValueError('The output of the "schedule" function '
                                 'should be float.')
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nbatch %05d: LearningRateScheduler reducing learning '
                      'rate to %s.' % (epoch + 1, lr))
    

#If we wanted to Evaluate on more than the macro AUC we could use this.             
#Evaluation function.
def Evalue_fun(pred,label):
    cla_to_keep=[]
    for j in range(0,Nclasses):
        cla_to_keep.append(np.mean(label[:,j])>0)

    test_L_eval=label[:,np.array(cla_to_keep)]
    pred_BOW_eval=pred[:,np.array(cla_to_keep)]
    
    macro_f1=f1_score(test_L_eval,np.where(pred_BOW_eval>0.2,1,0),average='macro')
    micro_f1=f1_score(test_L_eval,np.where(pred_BOW_eval>0.2,1,0),average='micro')
    macro_auc=roc_auc_score(test_L_eval,pred_BOW_eval,average='macro')
    micro_auc=roc_auc_score(test_L_eval,pred_BOW_eval,average='micro')
    
    return(macro_f1,micro_f1,macro_auc,micro_auc)


def create_model(key_dim,value_dim,atnheads,extract,LSTM_DIM,proj_dim,embed_dim,Nclasses,drop,embed_input_dim,Nlay,baseline,gpu,return_weights=False):     
        #set up Attention Model

        mula=Multi_Attention(feature_dim=key_dim,value_dim=value_dim,nheads=atnheads)
        
        if gpu ==True and tf.test.is_gpu_available():
            RNN=CuDNNLSTM
        else:
            RNN=LSTM
       
        #Which kind of extraction to use
        if extract=="lstm":
            def feat_ex(Input):

                feat=Bidirectional(RNN(LSTM_DIM, return_sequences=True))(Input)
                return feat

        if extract=="cnn_lstm":
            def feat_ex(Input):

                feat=Conv1D(LSTM_DIM,3,padding="same")(Input)
                feat=Bidirectional(RNN(LSTM_DIM, return_sequences=True))(feat)
                return feat

        if extract=="cnn":  
            def feat_ex(Input):

                feat=Conv1D(LSTM_DIM,3,padding="same")(Input)
                #feat=Bidirectional(CuDNNLSTM(LSTM_DIM, return_sequences=True))(feat)
                return feat

        #Defining a "Loop BLock" Projection, Normalization activation and droppout
        def Loop_Block(Input):
            lp=Loop_Projection(projection_dim=proj_dim)(Input)
            lp=BatchNormalization(axis=2)(lp)
            lp=Activation(activation="relu")(lp)
            lp=BatchNormalization(axis=1)(lp)
            lp=Dropout(drop)(lp)
            return lp

        #Lastly we bring it all together
        
        #define input
        model_input=keras.layers.Input(shape=[None])
        #embed
        emb=Embedding(embed_input_dim, embed_dim,trainable=True)(model_input)
        #get the extraction we use
        feat_e=feat_ex(emb)
        #droppout
        drop_feat=Dropout(drop)(feat_e)
        #Attention
        MA,weights=mula(drop_feat)
        #Loop Block, projection and normalization 
        if Nlay>1:
            ex=Loop_Block(MA)
            for j in range(Nlay-1):
                ex=Loop_Block(ex)
        else:
            ex=MA
        #output projection layer
        #ex=Loop_Projection(projection_dim=Nclasses,share_paramerters=True)(ex)
        ex=Permute((2,1))(ex)
        
        #we only allow words to positively contribute 
        out2=Dense(Nclasses)(ex)
        
        #sum the logits
        ex=sum_dum()(out2)
        #we use a global bias 
        #ex=global_bias()(ex)
        
        ex=Activation(activation="sigmoid")(ex)
        
        if return_weights == True:
            model=keras.Model(inputs=model_input,outputs=[ex,weights,out2])
        else:
            model=keras.Model(inputs=model_input,outputs=ex)

        sgd=optimizers.Adam(lr=0.0001,decay=0.000001)
        model.compile(optimizer=sgd,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        
        if baseline == True:
            model_input=keras.layers.Input(shape=[None])
            #embed
            emb=Embedding(len(word_tokenizer.word_index)+2, embed_dim,trainable=True)(model_input)
            feat=Conv1D(LSTM_DIM,3,padding="same")(emb)
            pool=GlobalMaxPool1D()(feat)
            fc= Dense(units=500,activation="relu")(pool)
            fc2= Dense(units=Nclasses,activation="sigmoid")(fc)
            model=keras.Model(inputs=model_input,outputs=fc2)
            sgd=optimizers.Adam(lr=0.0001,decay=0.000001)
            model.compile(optimizer=sgd,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        model.summary()
        return model 