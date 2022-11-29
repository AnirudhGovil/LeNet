import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader, Dataset

def convert(text):
    dic = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',
                        11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'} 
    return dic[text]

df=pd.read_csv("kaggle-az-handwritten-alphabets-in-csv-format/handwritten_data_785.csv").astype('float32')  # 28x28 images, same as MNIST

df.rename(columns={'0':'label'}, inplace=True)
x=df.drop('label',axis=1)
y=df['label']

data=df.copy()
data['mapped']=data['label'].apply(lambda df:convert(df))

x=shuffle(x)

xtrain,xtest,ytrain,ytest=train_test_split(x,y)

scalar=MinMaxScaler()
scalar.fit(xtrain)
xtrain=torch.tensor(scalar.transform(xtrain))
xtest=torch.tensor(scalar.transform(xtest))

ytrain=torch.tensor(ytrain.values)
ytest=torch.tensor(ytest.values)

trainset = (xtrain, ytrain)
testset = (xtest, ytest)

class AlphabetData(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        super().__init__()

    def __len__(self):
        return len(self.dataset[1])

    def __getitem__(self, index):
        return (torch.tensor(self.dataset[0][index]),torch.tensor(self.dataset[1][index]))

trainDataset = AlphabetData(trainset)
testDataset = AlphabetData(testset)

trainloader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testloader = DataLoader(testDataset, batch_size=64, shuffle=True)
