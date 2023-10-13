import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, classification_report
import pickle
import os


def preprocess_pandas(data, columns):
    df_ = pd.DataFrame(columns=columns)
    data['Sentence'] = data['Sentence'].str.lower()
    data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
    data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
    data['Sentence'] = data['Sentence'].str.replace('[^\w\s]','')                                                       # remove special characters
    data['Sentence'] = data['Sentence'].replace('\d', '', regex=True)                                                   # remove numbers
    for index, row in data.iterrows():
        word_tokens = word_tokenize(row['Sentence'])
        filtered_sent = [w for w in word_tokens if not w in stopwords.words('english')]
        df_ = df_.append({
            "index": row['index'],
            "Class": row['Class'],
            "Sentence": " ".join(filtered_sent[0:])
        }, ignore_index=True)
    return data

def accuracy(iterator): #get network accuracy on a dataset
    total = 0
    success = 0
    for batch_nr, (data, labels) in enumerate(iterator):
        pred = network(data)

        for p,label in zip(pred,labels):
            guess = torch.argmax(p, dim=-1)
            total+=1
            if guess.item() == label.item():
                success+=1

    return success/total    

# If this is the primary file that is executed (ie not an import of another file)
if __name__ == "__main__":

    # get data, pre-process and split
    
    dirname = os.path.dirname(os.path.abspath("ANN"))
    filename = os.path.join(dirname, 'testdataset.txt')

    data = pd.read_csv(filename, delimiter='\t', header=None)
    data.columns = ['Sentence', 'Class']
    data['index'] = data.index                                          # add new column index
    columns = ['index', 'Class', 'Sentence']
    data = preprocess_pandas(data, columns)                             # pre-process
    training_data, validation_data, training_labels, validation_labels = train_test_split( # split the data into training, validation, and test splits
        data['Sentence'].values.astype('U'),
        data['Class'].values.astype('int32'),
        test_size=0.10,
        random_state=0,
        shuffle=True
    )   
    # vectorize data using TFIDF and transform for PyTorch for scalability
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=50000, max_df=0.5, use_idf=True, norm='l2')
    training_data = word_vectorizer.fit_transform(training_data)        # transform texts to sparse matrix
    training_data = training_data.todense()                             # convert to dense matrix for Pytorch
    vocab_size = len(word_vectorizer.vocabulary_)
    #print(vocab_size)
    validation_data = word_vectorizer.transform(validation_data)
    validation_data = validation_data.todense()
    train_x_tensor = torch.from_numpy(np.array(training_data)).type(torch.FloatTensor)

    train_y_tensor = torch.from_numpy(np.array(training_labels)).long()
    validation_x_tensor = torch.from_numpy(np.array(validation_data)).type(torch.FloatTensor)
    validation_y_tensor = torch.from_numpy(np.array(validation_labels)).long()
    #fix dataset
    training_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    validation_dataset = TensorDataset(validation_x_tensor, validation_y_tensor)
    train_loader = DataLoader(training_dataset, batch_size=1, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    #network classifier
    
    '''softmax for probability for each option, can be used to implement logic for the bot in case the input is unclear'''
    network = nn.Sequential( 
    nn.Linear(vocab_size, 50),
    nn.ReLU(),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Linear(25, 6),
    nn.Softmax(dim=1) #axis
    )
    #hyperparameters
    optimizer = torch.optim.Adam(network.parameters(), lr = 0.01, weight_decay = 0.0001)
    loss_function = nn.CrossEntropyLoss() 
    epochs = 1
    #training
    
    def train(train_loader, validation_loader):
        t_losses=[]
        best_accuracy = 0

        for epoch in range(epochs):
            t_loss = 0
            for batch_nr, (data, label) in enumerate(train_loader):

                prediction = network(data)   
                loss = loss_function(prediction, label) 

                t_loss += loss.item()

                loss.backward()

                optimizer.step()

                optimizer.zero_grad()

                #Print the epoch, batch, and loss
                print(
                    '\rEpoch {} [{}/{}] - t_loss: {}'.format(
                        epoch, batch_nr+1, len(train_loader), loss
                    ),
                    end=''
                )
        # Calculate the best model        
            x = accuracy(validation_loader)
            if (x>best_accuracy):
                best_accuracy = x
                best_model = network


        print("\nDone!")
        return best_model

    network = train(train_loader, validation_loader)
    x=accuracy(validation_loader)
    print("Accuracy = "+str(x*100)+"%")

    dirname = os.path.dirname(os.path.abspath("ANN"))
    filename = os.path.join(dirname, 'network.pth')
    torch.save(network, filename)

    dirname = os.path.dirname(os.path.abspath("ANN"))
    filename = os.path.join(dirname, 'word_vectorizer.pickle')
    pickle.dump(word_vectorizer, open(filename, "wb"))
