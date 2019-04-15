# Medical Records ICD-9 Classification

In this project we developed a multi-attention model that is tailormade to the task of ICD classification. The main goal of this repository is to make training such a multi-attention model on new data as simple as possible. Using 4 different easy to understand notebooks. This implementation is based on keras and makes heavy use of tf.data and the tf.records file system. For a better understanding of the multi-attention model we are in the process of explaining it in an arxiv paper. 


## Installation

Clone the repository by running:

```
git clone https://github.com/PiSchool/noovle-open-source 
```
We recommend a seperate tensorflow conda environment. Because we are using the CUDNRNN tf.keras layer the GPU Version of Tensorflow has to be installed. All requiered packages can be installed by first activating the conda environment and then running: 

```
pip install -r requirements.txt
```



## ICD-9 Classification: code organization

Notebooks:
 - `Loading_Writing.ipynb`: This notebook turnes a new dataset consisting of texts and sequences of ICD-9 codes into TF records. To do that the following steps have to be done:
     1. On a first run Text and Label Tokenizer have to be initialized. A Tokenizer is a form of dictionary that turns sequences of texts and labels strings into sequences of integers such that it can be processed by a neural network. 
     2. Next we need data specific loading functions which return the texts and the labels as lists of strings. We include a loading functions for a mock dataset we provide along with this repo. We also include loading function for the open source MIMIC-III dataset (DOI: 10.1038/sdata.2016.35). MIMIC-III dataset is not included in the repo but access to it can be obtained (https://mimic.physionet.org/). 
     3. Then we call the update_and_write function. This function has as input a label and text tokenizer and the new texts and labels. The output will be 3 TF Records files for training, test and validation respectively. The update_and_write function does the following things: 
          - It cleans the texts. 
          - It searchs the text for new terms and adds them to the text Tokenizer 
          - It searches the labels for new terms and adds them to the label Tokenizer
          - It Encodes the labels and texts into integer sequences. 
          - It writes the integer sequences as tf records.
            
 - `Training.ipynb`: Contains the call to the training function: N.B. modify Hyper_fit options as needed should be called from shell. It also contains useful information regarding Tensorboard. The main idea is that we supply a set of tf records and receive a trained neural network all in one  function. The function does the following thing.
     1. It loads supplied tokenizers
     2. It sets up TensorFlow datasets from the supplied TF records
     3. It sets up a model function that creates a Neural Network given a set of hyperparameters
     4. Then, given on user specification it either trains a Neural Network a single time using supplied hyperparameters or it searches for optimal hyperparameters using gaussian processes.
     
 - `Inference_demo.ipynb`: Performs final inference. Either to predict on a new string or to predict on a text from our dataset to compare predictions to labels.
 
Python code:

 -  `Clean_function.py`: cleans the incoming text, removes numbers, spaces, brakets, punctuation.
    
 - `mimic_data.py`: load MIMIC-III or the mock dataset notes and labels.

 - `Fit_Function.py`: It contains the main hyperparameter optimization function used in the Training notebook.
    
 - `Keras_Layers.py`: Where all the custom layers are defined
        
 - `update_and_write.py`: Creates tf records of the text and labels dataset after splitting it into train validation and test set.
 
 - `evaluation_functions.py`: Helper functions for calculating various metrics. 
 


## Other files

The repository also contains a few auxiliary files:

 -  This README, containing all instructions for installing and running the project.

 -  The LICENSE under which the files are released.

 -  A requirements.txt file containing a full specification of the libraries used in the project.
    
    
## Authors
This project was developed by [Leander Loew](https://github.com/lysecret2) and [Sabato Leo](https://github.com/sabatoleo) during [Pi School's AI programme](http://picampus-school.com/programme/school-of-ai/) in Summer 2018. We want to thank Noovle for sponsoring our seats in the programm. 

