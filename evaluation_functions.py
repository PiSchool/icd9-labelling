import h5py
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score,f1_score,log_loss,precision_score,recall_score,roc_curve
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt

#Calculates the Macro and Weighted AUC, only on Classes present in the test set (else we get an error)
def auc_eval(pred,mode):
    label= test_labels
    cla_to_keep=[]
    for j in range(0,Nclasses):
        cla_to_keep.append(np.mean(label[:,j])>0)

    test_L_eval=label[:,np.array(cla_to_keep)]
    pred_BOW_eval=pred[:,np.array(cla_to_keep)]
    return roc_auc_score(test_L_eval,pred_BOW_eval,average=mode)

#Calculates the recall at n 
def recall_at_n(p,test_labels,n=5):
    prec_list=[]
    recall_list=[]

    for case in range(p.shape[0]):
        top_n=np.argsort(p[case,])[-n:]
        true_n=np.where(test_labels[case,])[0]
        
        recal_n=len(list(set(true_n) & set(top_n))) / len(true_n)
        precision_n=len(list(set(true_n) & set(top_n))) / len(top_n)

        recall_list.append(recal_n)
        prec_list.append(precision_n)
    return np.mean(recall_list),np.mean(prec_list)

#calculates multipe recall values at different n
def recall_time_series(p):
    ts_list=[]
    for j in range(1,30):
        ts_list.append(recall_at_n(p,test_labels,j))
    return ts_list
    
#Plots histogram for recall
def plot_hist_recall(p,n=3):
    recall_list=[]

    for case in range(p.shape[0]):
        top_n=np.argsort(p[case,])[-n:]
        true_n=np.where(test_labels[case,])[0]
        recal_n=len(list(set(true_n) & set(top_n))) / len(true_n)
        
        recall_list.append(recal_n)

    plt.hist(recall_list,100, alpha=0.5, color = "red")
    plt.xlabel('Recall at {}'.format(n))
  
#plots hostogram for precision
def plot_hist_precision(p,n=3):
    prec_list=[]

    for case in range(p.shape[0]):
        top_n=np.argsort(p[case,])[-n:]
        true_n=np.where(test_labels[case,])[0]
        
        precision_n=len(list(set(true_n) & set(top_n))) / len(top_n)

        prec_list.append(precision_n)
        
    plt.hist(prec_list,100, alpha=0.5, color = "black")
    plt.xlabel('Precision at {}'.format(n))

def auc_histogram_overlay(auc1=None,auc2=None,bins=100, legend_loc='upper left'):
    SMALL_SIZE = 15
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 15
    plt.hold(True)
    plt.hist(auc1, bins, alpha=0.5, label='model1',density=True)
    plt.hist(auc2, bins, alpha=0.5, label='model2',density=True)
    plt.legend(loc=legend_loc,fontsize = 'xx-large')
    plt.xlabel('AUC score')
    plt.ylabel('Entries')
    #plt.rc('axes', labelsize=30)
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=50)     # fontsize of the axes title
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.savefig('./overley_hist.png')
    p = plt.show()
    return p



