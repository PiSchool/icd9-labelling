#The general use of this is to get TF Records and do Hyperparameter Optimization to get a well trained model in return. 
#we use sckit optimize for hyperparameter optiization and Keras for model training


import tensorflow as tf
import functools

import matplotlib.pyplot as plt
import numpy as np
import math

import numpy as np
import pandas as pd
import pickle,pickle
import math

import random

from tensorflow import keras
from tensorflow.python.keras import Model,optimizers,initializers
from tensorflow.python.keras.layers import Input,Conv1D,Embedding,Bidirectional,Dense,TimeDistributed,Permute,Activation,Add,Dot,Dropout,Lambda,GlobalMaxPool1D,GlobalAveragePooling1D,BatchNormalization,Convolution1D
from tensorflow.python.keras import backend as K

if tf.test.is_gpu_available():
    from tensorflow.python.keras.layers import CuDNNLSTM
    
from tensorflow.python.keras.layers import LSTM
import h5py
import logging
from sklearn.metrics import roc_auc_score
best_auc  = 0
from tensorflow.python.keras.callbacks import Callback, EarlyStopping


from Keras_Layers import Loop_Projection,Multi_Attention,sum_dum,LearningRateScheduler,create_model

#skopt functions
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.utils import use_named_args

#tensorboard
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import load_model

def Hyper_fit(path_best_model='best_model.keras',TB_logs='logs',steps=30000+50000,validation_steps=0.1*30000,train_files=None,validation_files=None,rounds=12, label_tokenizer_path=None,text_tokenizer_path=None,max_epoch=1,buffer_size=100000,default_parameters=[88,2,64,128,128,32,128,0.2,50,'lstm'],batch_size=128,evaluation="macro",single_fit=False,baseline=False,interval=10,wait_epochs=0,eval_size=100,save_cpu=True):
    '''
    path_best_model: is the model path for the best model 
    TB_logs: main for folder for tensorboard
    steps: is the number of observations in the training set
    validation_steps: is the number of observations in the validation set
    train_files: a tf record or a list of tf records of training files
    validation_files: a tf record or a list of tf records of vaidation files
    rounds: The number of Hyperparameter Rounds aka different models we want to train, each round is about 2-4 hours
    label_tokenizer_path,text_tokenizer_path: the Tokenizer we want to use for the current training run
    max_epoch: the number of epochs we want to train each model , we still do early stopping
    buffer_size: should be set to training size for best performance, however has some impact on training duration
    default_parameters a set of default parameters
    batch_size: the size of each training batch
    evaluation: The kind of AUC we want to compute one of Micro, Macro or Weighted
    single_fit: If we just want to have a single fit, no hyperparameter tuning
    baseline: If we want to train a baseline model, only use with singe fit. 
    
    
    '''
    
    #Loading Tokenizer
    label_tokenizer = pickle.load(open(label_tokenizer_path, 'rb'))
    word_tokenizer = pickle.load(open(text_tokenizer_path, 'rb'))
    
    #Has to be +1 because of how we set up the tokenizer
    Nclasses=len(label_tokenizer.word_index)+1
    
    #here we have to create many of functions which depend on the Nclasses, which depends on loading the tokenizer so we have to define them here sadly. 


    #Evaluation function.
    def Evalue_fun_simple(pred,label):
        
        cla_to_keep=[]
        for j in range(0,pred.shape[1]):
            cla_to_keep.append(np.mean(label[:,j])>0)

        cla_to_keep=np.where(cla_to_keep)
        cla_to_keep=cla_to_keep[0]
        if eval_size is not None:
            cla_to_keep=np.array(random.sample(list(cla_to_keep),eval_size))

        test_L_eval=label[:,np.array(cla_to_keep)]
        pred_BOW_eval=pred[:,np.array(cla_to_keep)]
        #This takes forever
        macro_auc=roc_auc_score(test_L_eval,pred_BOW_eval,average=evaluation) 

        return(macro_auc)

    #We will creae directories depending on the Hyperparams, this is the way Tensorboard likes it
    def log_dir_name(dim_lr_dim,
              dim_Nlay,
              dim_LSTM_DIM,
              dim_key_dim,
              dim_value_dim,
              dim_atnheads,
              dim_proj_dim,
              dim_drop,
              dim_embed_dim,
              dim_extract
              ):

        # The dir-name for the TensorBoard log-dir.
        s = "./"+ TB_logs+"/lr_{}_Nlay_{}_LSTM_{}_key_{}_value_{}_atnheads_{}_proj_dim_{}_drop_{}_embed_dim_{}_extract{}/"

        # Insert all the hyper-parameters in the dir-name.
        log_dir = s.format(dim_lr_dim,

                  dim_Nlay,
                  dim_LSTM_DIM,
                  dim_key_dim,
                  dim_value_dim,
                  dim_atnheads,
                  dim_proj_dim,
                  dim_drop,
                  dim_embed_dim,
                  dim_extract
                  )

        return log_dir

    
    #We have to define what we expect from the TFRECORDS
    def parse_txt_sequence(record):
        '''
        Returns:
            token_indexes: sequence of token indexes of words present in document
            label_indexes: sequence of token indexes of labels present in document
            
        '''
        sequence_features = {
            'token_indexes': tf.FixedLenSequenceFeature([], dtype=tf.int64),
            'label_indexes': tf.FixedLenSequenceFeature([], dtype=tf.int64),
            }

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(record, 
            sequence_features=sequence_features)

        #Here it is not a batch but it is a single example.
        #super col: THis onehot encodes in one line of super efficient tf code!

        label=tf.reduce_sum(tf.one_hot(sequence_parsed['label_indexes'],depth=Nclasses),axis=0)
       
        #we might have multiple values of the same code, we hande it here. 
        label=tf.clip_by_value(label,clip_value_max=1,clip_value_min=0)
        return (sequence_parsed['token_indexes'],label)

    
    #this loads the validation dataset to memory. 
    def load_to_memory(tfrec=None):

        #Argument the tf records file we want to load to memory. 
        valid_dataset = tf.data.TFRecordDataset(tfrec)
        valid_dataset = valid_dataset.map(parse_txt_sequence)
        #kind of a trick to load the whole valiation dataset. 
        
        #TODO INCREASE 1000 or find better way
        
        valid_dataset = valid_dataset.padded_batch(10000, padded_shapes=([None],[Nclasses]))
        #this way we ge tis as a numpy array
        iterator = valid_dataset.make_one_shot_iterator()
        next_element=iterator.get_next()
        sess = tf.Session()
        Val_S,Val_L=sess.run(next_element)
        sess.close()
        return Val_S,Val_L

    
    #A callback function that calculates the AUC every interval and does Early stoppping when the AUC starts decreasing. 
    class IntervalEvaluation(Callback):
        def __init__(self, validation_data=(), interval=1):
            super(Callback, self).__init__()

            self.interval = interval
            self.X_val, self.y_val = validation_data
            self.score_list=[0]


        def on_epoch_end(self, epoch, logs={}):
            if epoch % self.interval == 0 and epoch>wait_epochs:
                y_pred = self.model.predict(self.X_val, verbose=0)
                score= Evalue_fun_simple(y_pred,self.y_val)
                self.score_list.append(score)
                print(score)
                #We leave 5 epochs always 
                #We stop when AUC starts decreasing
                if (np.max(np.asarray(self.score_list))-score)>0.0001:
                    self.model.stop_training = True
                    

    #Hyperparameters 
    #[88,1,100,128,128,30,128,0.2,50,'lstm']
    

    #For Each parameter we have to define the ranges important if we set Initial Parameters out of the ranges we get an error CAREFULL INITIAL VALUE M IN THE RANGES COMMOM ERROR
    dim_lr_dim = Integer(low=10, high=150,name='lr_dim') # THis relates to the Learning Rate we set, (it isnt directly the LR)
    dim_Nlay = Integer(low=1, high=4,name='Nlay') #Number of Layer after the aggregation
    dim_LSTM_DIM = Integer(low=32, high=128,name='LSTM_DIM') #The dimension of the feature extractor ( also used for the CNN )
    dim_key_dim=Integer(low=32, high=512,name='key_dim') #THe dimension of the Key we use for attention
    dim_value_dim=Integer(low=32, high=512,name='value_dim') #The dimension for the Values we use for attention
    dim_atnheads=Integer(low=2, high=60,name='atnheads') #The number of Attention heads 
    dim_proj_dim=Integer(low=32, high=300,name='proj_dim') #The "projection" dimmension
    dim_drop  = Real(low=float(0), high=float(0.9), prior='uniform',name='drop') #THe Droppout we apply
    dim_embed_dim = Integer(low=30, high=150,name='embed_dim') #The dimension we embed to 
    dim_extract= Categorical(categories=['lstm','cnn_lstm','cnn'],name='extract') #The Feture extraaction mwthod

    #Collect them in a list
    dimensions = [dim_lr_dim,
                  dim_Nlay,
                  dim_LSTM_DIM,
                  dim_key_dim,
                  dim_value_dim,
                  dim_atnheads,
                  dim_proj_dim,
                  dim_drop,
                  dim_embed_dim,
                  dim_extract
                 ]

   



#this is the main fit function. It takes Hyperparameters as input, sets up and trains a model and returns th best score from this model. 

    @use_named_args(dimensions=dimensions)
    def fit_model(lr_dim,Nlay,LSTM_DIM,key_dim,value_dim,atnheads,proj_dim,drop,embed_dim ,extract
                  ):


        train_steps=round((steps)/batch_size)

        #Getting Data

        #Get Training Dataset 
        train_dataset = tf.data.TFRecordDataset(train_files)
        train_dataset = train_dataset.map(parse_txt_sequence).shuffle(buffer_size=buffer_size)
        train_dataset = train_dataset.padded_batch(batch_size, padded_shapes=([None],[Nclasses]))
        train_dataset=train_dataset.repeat()

        # Create validation dataset from TFRecords
        valid_dataset = tf.data.TFRecordDataset(validation_files)
        valid_dataset = valid_dataset.map(parse_txt_sequence)
        valid_dataset = valid_dataset.padded_batch(batch_size, padded_shapes=([None],[Nclasses]))
        valid_dataset=valid_dataset.repeat()

        
        #Load validation sets to memory
        Val_A=load_to_memory(validation_files)
        eval_A=IntervalEvaluation(validation_data=Val_A,interval=interval)

        
        #LearningRateScheduler
        def decay(epoch):
            lrate = (lr_dim**-.5) * min((epoch+1)**-.5,(epoch+1)*2000**-1.5)
            
            return lrate
        lrate=LearningRateScheduler(decay)
               
        #We put the model creation in a seperat file, cleaner this way 
        model=create_model(key_dim,value_dim,atnheads,extract,LSTM_DIM,proj_dim,
                               embed_dim,Nclasses,drop,len(word_tokenizer.word_index)+1,Nlay,baseline,True)
        
        #We create different Log dirs for tensorboard 
        log_dir = log_dir_name(lr_dim,Nlay,LSTM_DIM,key_dim,value_dim,atnheads,proj_dim,drop,embed_dim,extract)
        callback_log = TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            batch_size=32,
            write_graph=True,
            write_grads=False,
            write_images=False)

        #We collect all different Callbacks ( learning rate scheudler, the "intervall evlauation of the AUC" and the logging for tensorbaord
        cblist=[lrate,eval_A,callback_log]
        #we do a try here because, depending on the input we might get too big a model to train, but then we dont really care because it 
        #would be a bad to do inference on anyways so we skip that model and return a bad AUC. 
        #try:
        history=model.fit(train_dataset, epochs=max_epoch, steps_per_epoch=train_steps, validation_data=valid_dataset,validation_steps=round(validation_steps/batch_size),callbacks=cblist)
            
        #except:
        #print("Can't fit model, OOM try reducing the Batch Size or a smaller model.") 
        final_result=max(eval_A.score_list)   
        print(final_result)
        #get the best auc variable
        global best_auc
        #This was a rare exception once, we should remove that part. 
        if final_result > best_auc:
            # Save the new model to disk.
                #if save_cpu==True:
                #First we save the weights 
                model.save_weights(path_best_model)
                #Then we replace the model by the same modle but on CPU and tell it to return the weightss
                model=create_model(key_dim,value_dim,atnheads,extract,LSTM_DIM,proj_dim,
                                   embed_dim,Nclasses,drop,len(word_tokenizer.word_index)+1,Nlay,baseline,False,True)
                #Then we load the GPU model's weights in ot 
                model.load_weights(path_best_model)

                #In the end we save the actual modle 
                model.save(path_best_model)
                best_auc = final_result
                df =pd.DataFrame(np.asarray([best_auc,lr_dim,batch_size,Nlay,LSTM_DIM,key_dim,value_dim,atnheads,proj_dim,drop,embed_dim ,extract]))
                df.to_csv("best_AUC_model.csv")

            # Update the classification accuracy.

        del model
        K.clear_session()

        return -final_result
    
    #do the whole fit
    if single_fit== False:
        search_result = gp_minimize(func=fit_model,
                                    dimensions=dimensions,
                                    acq_func='EI', # Expected Improvement.
                                    n_calls=rounds,
                                    verbose=True,
                                    x0=default_parameters)

        plot_convergence(search_result)
        search_result.x
        print(best_auc)
    
    if single_fit== True:
        dimensions=default_parameters
        lr_dim,Nlay,LSTM_DIM,key_dim,value_dim,atnheads,proj_dim,drop,embed_dim ,extract=default_parameters
        fit_model(default_parameters)
    
