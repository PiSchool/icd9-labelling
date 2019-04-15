#load packages
import pandas as pd
import numpy as np
import re

def load_merge(diag_path='./DIAGNOSES_ICD.csv', notes_path='./NOTEEVENTS.csv'):

    df_diag = pd.read_csv(diag_path)
    df_notes = pd.read_csv(notes_path)
    
    df_diag=df_diag.dropna()
    
    #group the ICD9 codes by Admission ID
    df_diag = df_diag.groupby(('HADM_ID'))
    df_diag = df_diag['ICD9_CODE'].unique()
    df_diag = df_diag.reset_index()
    
    #select only discharge summary and report from the notes
    df_notes = df_notes.loc[df_notes['CATEGORY'] == 'Discharge summary']
    df_notes = df_notes.loc[df_notes['DESCRIPTION'] == 'Report']
    
    #merge notes and diagnosis by Admission ID (HADM_ID) 
    # -- Caveat, there is multiple texts associate to a single HADM_ID, we will have some noisy labels this way
    merge = pd.merge(df_notes, df_diag, on=['HADM_ID'], how='left')
    #select only TEXT and ICD9_CODE columns
    merge = merge[['TEXT','ICD9_CODE']]
    merge=merge.dropna()
    print('Total dataset has ',len(merge), 'entries')
    #Caveat:there is ca 3000 multiple texts associate to a single HADM_ID

    unclean_Text = merge.TEXT
    labels=merge.ICD9_CODE
    labels=labels.values
    labels=[x.tolist() for x in labels.tolist()]
    
    return unclean_Text, labels

def load_dummy(file='data/fake_dataset.csv'):
    dat=pd.read_csv(file)
    df_out = dat.assign(Fake_labels=dat.Fake_labels.str.strip('[]').str.split(','))
    texts=df_out.Fake_Text.tolist()
    labels=df_out.Fake_labels.tolist()
    labels=np.asarray([[re.sub("[^0-9]", "",x) for x in i] for i in labels])
    return texts,labels