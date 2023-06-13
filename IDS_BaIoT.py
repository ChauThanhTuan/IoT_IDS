import torch
import numpy as np
import pandas as pd

from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import MaxPool1d
from torch.nn import Dropout
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import BatchNorm1d
from torch import flatten
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


BATCH_SIZE = 850
EPOCHS = 50
LOG_INTERVAL = 100
lr = 0.00005


early_stop_thresh = 5
best_accuracy = -1
best_epoch = -1

class Net(Module):
    def __init__(self, numChannels, classes):
        # call the parent constructor
        super(Net, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = Conv1d(in_channels=numChannels, out_channels=150,
        	kernel_size=1)
        self.batchnorm1 = BatchNorm1d(150)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool1d(kernel_size=1)
        self.dropout1 = Dropout(p=0.1)
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = Conv1d(in_channels=150, out_channels=180,
        	kernel_size=1)
        self.batchnorm2 = BatchNorm1d(180)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool1d(kernel_size=1)
        self.dropout2 = Dropout(p=0.1)
        # initialize first set of FC => RELU layers
        self.fc1 = Linear(in_features=180, out_features=60)
        self.batchnorm4 = BatchNorm1d(60)
        self.relu4 = ReLU()
        self.dropout4 = Dropout(p=0.1)
        # initialize first set of FC => RELU layers
        self.fc2 = Linear(in_features=60, out_features=40)
        self.batchnorm5 = BatchNorm1d(40)
        self.relu5 = ReLU()
        self.dropout5 = Dropout(p=0.1)
        # initialize our softmax classifier
        self.fc3 = Linear(in_features=40, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)
  
    def forward(self, x):
        # data = data[..., np.newaxis]
        x = x.reshape(x.shape[0], x.shape[1], 1)

        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        # flatten the output from the previous layer and pass it
        # through our set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.batchnorm4(x)
        x = self.relu4(x)
        # x = self.dropout4(x)
        x = self.fc2(x)
        x = self.batchnorm5(x)
        x = self.relu5(x)
        # x = self.dropout5(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc3(x)
        output = self.logSoftmax(x)
        # return the output predictions
        return output



def ReadDataFromCSV(benign, g_c, g_j, g_s, g_u, m_a, m_sc, m_sy, m_u, m_u_p, isTest=False):

    benign=pd.read_csv(benign)
    g_c=pd.read_csv(g_c)
    g_j=pd.read_csv(g_j)
    g_s=pd.read_csv(g_s)
    g_u=pd.read_csv(g_u)
    m_a=pd.read_csv(m_a)
    m_sc=pd.read_csv(m_sc)
    m_sy=pd.read_csv(m_sy)
    m_u=pd.read_csv(m_u)
    m_u_p=pd.read_csv(m_u_p)

    if isTest:
        benign=benign.sample(frac=1,replace=False)
        g_c=g_c.sample(frac=1,replace=False)
        g_j=g_j.sample(frac=1,replace=False)
        g_s=g_s.sample(frac=1,replace=False)
        g_u=g_u.sample(frac=1,replace=False)
        m_a=m_a.sample(frac=1,replace=False)
        m_sc=m_sc.sample(frac=1,replace=False)
        m_sy=m_sy.sample(frac=1,replace=False)
        m_u=m_u.sample(frac=1,replace=False)
        m_u_p=m_u_p.sample(frac=1,replace=False)
    else:
        benign=benign.sample(frac=0.55,replace=False)
        g_c=g_c.sample(frac=0.5,replace=False)
        g_j=g_j.sample(frac=1,replace=False)
        g_s=g_s.sample(frac=1,replace=False)
        g_u=g_u.sample(frac=0.28,replace=False)
        m_a=m_a.sample(frac=0.31,replace=False)
        m_sc=m_sc.sample(frac=0.37,replace=False)
        m_sy=m_sy.sample(frac=0.27,replace=False)
        m_u=m_u.sample(frac=0.16,replace=False)
        m_u_p=m_u_p.sample(frac=0.38,replace=False)

    benign['type']='benign'
    m_u['type']='mirai_udp'
    g_c['type']='gafgyt_combo'
    g_j['type']='gafgyt_junk'
    g_s['type']='gafgyt_scan'
    g_u['type']='gafgyt_udp'
    m_a['type']='mirai_ack'
    m_sc['type']='mirai_scan'
    m_sy['type']='mirai_syn'
    m_u_p['type']='mirai_udpplain'

    data=pd.concat([benign,m_u,g_c,g_j,g_s,g_u,m_a,m_sc,m_sy,m_u_p],
                axis=0, sort=False, ignore_index=True)
    return data


datatrain = ReadDataFromCSV('./Datasets/N_BaIoT/1.benign.csv', './Datasets/N_BaIoT/1.gafgyt.combo.csv', './Datasets/N_BaIoT/1.gafgyt.junk.csv', './Datasets/N_BaIoT/1.gafgyt.scan.csv', './Datasets/N_BaIoT/1.gafgyt.udp.csv', './Datasets/N_BaIoT/1.mirai.ack.csv', './Datasets/N_BaIoT/1.mirai.scan.csv', './Datasets/N_BaIoT/1.mirai.syn.csv', './Datasets/N_BaIoT/1.mirai.udp.csv', './Datasets/N_BaIoT/1.mirai.udpplain.csv')
datatest = ReadDataFromCSV('./Datasets/N_BaIoT/2.benign.csv', './Datasets/N_BaIoT/2.gafgyt.combo.csv', './Datasets/N_BaIoT/2.gafgyt.junk.csv', './Datasets/N_BaIoT/2.gafgyt.scan.csv', './Datasets/N_BaIoT/2.gafgyt.udp.csv', './Datasets/N_BaIoT/2.mirai.ack.csv', './Datasets/N_BaIoT/2.mirai.scan.csv', './Datasets/N_BaIoT/2.mirai.syn.csv', './Datasets/N_BaIoT/2.mirai.udp.csv', './Datasets/N_BaIoT/2.mirai.udpplain.csv', isTest=True)
datatrain = datatrain.append(ReadDataFromCSV('./Datasets/N_BaIoT/2.benign.csv', './Datasets/N_BaIoT/2.gafgyt.combo.csv', './Datasets/N_BaIoT/2.gafgyt.junk.csv', './Datasets/N_BaIoT/2.gafgyt.scan.csv', './Datasets/N_BaIoT/2.gafgyt.udp.csv', './Datasets/N_BaIoT/2.mirai.ack.csv', './Datasets/N_BaIoT/2.mirai.scan.csv', './Datasets/N_BaIoT/2.mirai.syn.csv', './Datasets/N_BaIoT/2.mirai.udp.csv', './Datasets/N_BaIoT/2.mirai.udpplain.csv'), ignore_index = True)
datatrain = datatrain.append(ReadDataFromCSV('./Datasets/N_BaIoT/4.benign.csv', './Datasets/N_BaIoT/4.gafgyt.combo.csv', './Datasets/N_BaIoT/4.gafgyt.junk.csv', './Datasets/N_BaIoT/4.gafgyt.scan.csv', './Datasets/N_BaIoT/4.gafgyt.udp.csv', './Datasets/N_BaIoT/4.mirai.ack.csv', './Datasets/N_BaIoT/4.mirai.scan.csv', './Datasets/N_BaIoT/4.mirai.syn.csv', './Datasets/N_BaIoT/4.mirai.udp.csv', './Datasets/N_BaIoT/4.mirai.udpplain.csv'), ignore_index = True)
datatrain = datatrain.append(ReadDataFromCSV('./Datasets/N_BaIoT/5.benign.csv', './Datasets/N_BaIoT/5.gafgyt.combo.csv', './Datasets/N_BaIoT/5.gafgyt.junk.csv', './Datasets/N_BaIoT/5.gafgyt.scan.csv', './Datasets/N_BaIoT/5.gafgyt.udp.csv', './Datasets/N_BaIoT/5.mirai.ack.csv', './Datasets/N_BaIoT/5.mirai.scan.csv', './Datasets/N_BaIoT/5.mirai.syn.csv', './Datasets/N_BaIoT/5.mirai.udp.csv', './Datasets/N_BaIoT/5.mirai.udpplain.csv'), ignore_index = True)
datatrain = datatrain.append(ReadDataFromCSV('./Datasets/N_BaIoT/8.benign.csv', './Datasets/N_BaIoT/8.gafgyt.combo.csv', './Datasets/N_BaIoT/8.gafgyt.junk.csv', './Datasets/N_BaIoT/8.gafgyt.scan.csv', './Datasets/N_BaIoT/8.gafgyt.udp.csv', './Datasets/N_BaIoT/8.mirai.ack.csv', './Datasets/N_BaIoT/8.mirai.scan.csv', './Datasets/N_BaIoT/8.mirai.syn.csv', './Datasets/N_BaIoT/8.mirai.udp.csv', './Datasets/N_BaIoT/8.mirai.udpplain.csv'), ignore_index = True)
datatrain = datatrain.append(ReadDataFromCSV('./Datasets/N_BaIoT/9.benign.csv', './Datasets/N_BaIoT/9.gafgyt.combo.csv', './Datasets/N_BaIoT/9.gafgyt.junk.csv', './Datasets/N_BaIoT/9.gafgyt.scan.csv', './Datasets/N_BaIoT/9.gafgyt.udp.csv', './Datasets/N_BaIoT/9.mirai.ack.csv', './Datasets/N_BaIoT/9.mirai.scan.csv', './Datasets/N_BaIoT/9.mirai.syn.csv', './Datasets/N_BaIoT/9.mirai.udp.csv', './Datasets/N_BaIoT/9.mirai.udpplain.csv'), ignore_index = True)

def DataPreprocessing(data, num, BATCH_SIZE):
    #how many instances of each class
    data.groupby('type')['type'].count()

    #shuffle rows of dataframe 
    sampler=np.random.permutation(len(data))
    data=data.take(sampler)
    data = data[:num]

    threat_types = data["type"].values
    encoder = LabelEncoder()
    # use LabelEncoder to encode the threat types in numeric values
    y = encoder.fit_transform(threat_types)
    print("Shape of target vector : ", y.shape)

    #drop labels from training dataset
    data=data.drop(columns='type')

    #standardize numerical columns
    def standardize(df,col):
        df[col]= (df[col]-df[col].mean())/df[col].std()

    data_st=data.copy()
    for i in (data_st.iloc[:,:-1].columns):
        standardize(data_st,i)

    X = data_st.values

    # Create pytorch tensor from X, y
    test_inputs = torch.tensor(X,dtype=torch.float)
    test_labels = torch.tensor(y).type(torch.LongTensor)
    dataset = TensorDataset(test_inputs, test_labels)
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return test_inputs, test_labels, data_loader

_, _, train_data_loader = DataPreprocessing(datatrain, 1700000, BATCH_SIZE)
X_test, y_test, test_data_loader = DataPreprocessing(datatest, 500000, BATCH_SIZE)
n_feature = X_test.shape[1]
n_class = np.unique(y_test).shape[0]

print("Number of testing features : ", n_feature)
print("Number of testing classes : ", n_class)

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def train(model, train_data_loader, test_data_loader, optimizer, EPOCHS, early_stop_thresh, best_accuracy, best_epoch):
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Iterate through each gateway's dataset
        for idx, (data, target) in enumerate(train_data_loader):
            batch_idx = idx + 1            

            # Clear previous gradients (if they exist)
            optimizer.zero_grad()
            # Make a prediction
            output = model(data)
            # Calculate the cross entropy loss [We are doing classification]
            loss = F.cross_entropy(output, target)
            # Calculate the gradients
            loss.backward()
            # Update the model weights
            optimizer.step()

            if batch_idx != 0 and batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\nLoss: {:.6f}\n'.format(
                    epoch, batch_idx * BATCH_SIZE, len(train_data_loader) * BATCH_SIZE,
                    100. * batch_idx / len(train_data_loader), loss.item()))
        
        acc = test(model, test_data_loader, epoch)

        if acc > best_accuracy + 1:
            best_accuracy = acc
            best_epoch = epoch
            print("best_accuracy: ", best_accuracy)
            print("best_epoch: ", best_epoch)
            checkpoint(model, "IoT_Intrusions_Detection.pth")
        elif epoch - best_epoch > early_stop_thresh:
            print("Early stopped training at epoch %d" % epoch)
            break  # terminate the training loop
 
    resume(model, "IoT_Intrusions_Detection.pth")

    return model

def test(model, test_data_loader, epoch):
    model.eval()
    correct = 0
    total = 0

    for (data, target) in test_data_loader:
        # Make a prediction
        output = model(data)
        # Get the model back from the gateway
        # Calculate the cross entropy loss
        loss = F.cross_entropy(output, target)
        # Get the index of the max log-probability 
        _, pred = torch.max(output.data, 1)
        # Get the number of instances correctly predicted
        # correct += pred.eq(target.view_as(pred)).sum()
        total += target.size(0)
        correct += (target == pred).sum().item()
    
    # get the loss back
    loss = loss
    acc = 100. * correct / total

    print('Test set epoch {}: Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        int(epoch), loss.item(), correct, total, acc))
    
    return acc

model = Net(n_feature, n_class)
optimizer = optim.Adam(model.parameters(), lr=lr)


model = train(model, train_data_loader, test_data_loader, optimizer, EPOCHS, early_stop_thresh, best_accuracy, best_epoch)

# # Save the model
# torch.save(model.state_dict(), "IoT_Intrusions_Detection.pth")

# Reload the model in a new model object
model_new = Net(n_feature,n_class)
model_new.load_state_dict(torch.load("IoT_Intrusions_Detection.pth"))
model_new.eval()

y_true = y_test
y_pred = []

for item in X_test:
    item = item.reshape(1, n_feature, 1)
    pred = model_new(item)
    pred = int(pred.argmax().data.numpy())
    y_pred.append(pred)
target_names = ['benign', 'gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_udp', 'mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udp', 'mirai_udpplain']
print(classification_report(y_true, y_pred, target_names=target_names))


