import socket
import pickle
import numpy as np
import pandas as pd
import threading
from threading import Thread
import torch

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


BATCH_SIZE = 150
EPOCHS = 100
LOG_INTERVAL = 200
lr = 0.0005

class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
    
class Worker():
    
    def __init__(self):
        super().__init__()

    def create_socket(self, *args):
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
        print("Socket Created")

    def connect(self, *args):
        try:
            ip = "localhost"
            port = 22334
            self.soc.connect((ip, int(port)))
            print("Successful Connection to the Server")
    
        except BaseException as e:
            print(f"Error Connecting to the Server: {e}")

    def recv_train_model(self, *args):
        # global keras_ga

        recvThread = RecvThread(worker=self, buffer_size=1024, recv_timeout=3600)
        recvThread.start()

    def close_socket(self, *args):
        self.soc.close()
        print("Socket Closed")

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

    
class RecvThread(threading.Thread):

    def __init__(self, worker, buffer_size, recv_timeout):
        threading.Thread.__init__(self)
        self.worker = worker
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True:
            try:
                self.worker.soc.settimeout(self.recv_timeout)
                received_data += self.worker.soc.recv(self.buffer_size)

                try:
                    pickle.loads(received_data)
                    print("All data is received from the server.")
                    # If the previous pickle.loads() statement is passed, this means all the data is received.
                    # Thus, no need to continue the loop and a break statement should be excuted.
                    break
                except BaseException:
                    # An exception is expected when the data is not 100% received.
                    pass

            except socket.timeout:
                print(f"A socket.timeout exception occurred because the server did not send any data for {self.recv_timeout} seconds.")
                return None, 0
            except BaseException as e:
                print(f"Error While Receiving Data from the Server: {e}.")
                return None, 0

        try:
            received_data = pickle.loads(received_data)
        except BaseException as e:
            print(f"Error Decoding the Data: {e}.\n")
            return None, 0
        
        # print(received_data)
        return received_data, 1

    def run(self):
        global model
        
        isRequestModel = True

        early_stop_thresh = 3
        best_accuracy = -1
        best_epoch = -1 

        for epoch in range(EPOCHS + 1):
            if isRequestModel:
                isRequestModel = False
                data = {"subject": "echo", "data": "Request model"}
            else:
                data = {"subject": "model", "data": model}

            data_byte = pickle.dumps(data)

            print(f"Sending a Message of Type {data['subject']} to the Server")
            print("===============================================================================")
            print("epoch: ", epoch)
            try:
                self.worker.soc.sendall(data_byte)
            except BaseException as e:
                print("Error Connecting to the Server. The server might has been closed.")
                print(f"Error Connecting to the Server: {e}")
                break

            print("Receiving Reply from the Server")
            received_data, status = self.recv()
            if status == 0:
                print("Nothing Received from the Server")
                break
            else:
                print("New Message from the Server")

            subject = received_data["subject"]
            if subject == "model":
                model = received_data["data"]

                # create a thread for the function train
                train_thread = ThreadWithReturnValue(target=train, args=(model, train_data_loader, test_data_loader, epoch))
                # start the thread
                train_thread.start()
                # wait for the thread to finish
                model = train_thread.join()

                acc = test(model=model, test_data_loader=test_data_loader, epoch=epoch)
    
                if acc > best_accuracy + 1:
                    best_accuracy = acc
                    best_epoch = epoch
                    checkpoint(model, "IoT_Intrusions_Detection1.pt")
                elif epoch - best_epoch > early_stop_thresh:
                    print("Early stopped training at epoch %d" % epoch)
                    break  # terminate the training loop
    

            elif subject == "done":
                break
            else:
                print(f"Unrecognized Message Type: {subject}")
                return
        
        resume(model, "IoT_Intrusions_Detection1.pt")
        data = {"subject": "Done", "data": "Close connetion"}
        print(f"Sending a Message of Type {data['subject']} to the Server")
        
        data_byte = pickle.dumps(data)

        try:
            self.worker.soc.sendall(data_byte)
        except BaseException as e:
            print("Error Connecting to the Server. The server might has been closed.")
            print(f"Error Connecting to the Server: {e}")
        
        print("Model is Trained")
        worker.close_socket()


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


# datatrain = ReadDataFromCSV('./Datasets/N_BaIoT/1.benign.csv', './Datasets/N_BaIoT/1.gafgyt.combo.csv', './Datasets/N_BaIoT/1.gafgyt.junk.csv', './Datasets/N_BaIoT/1.gafgyt.scan.csv', './Datasets/N_BaIoT/1.gafgyt.udp.csv', './Datasets/N_BaIoT/1.mirai.ack.csv', './Datasets/N_BaIoT/1.mirai.scan.csv', './Datasets/N_BaIoT/1.mirai.syn.csv', './Datasets/N_BaIoT/1.mirai.udp.csv', './Datasets/N_BaIoT/1.mirai.udpplain.csv')
# datatrain = datatrain.append(ReadDataFromCSV('./Datasets/N_BaIoT/4.benign.csv', './Datasets/N_BaIoT/4.gafgyt.combo.csv', './Datasets/N_BaIoT/4.gafgyt.junk.csv', './Datasets/N_BaIoT/4.gafgyt.scan.csv', './Datasets/N_BaIoT/4.gafgyt.udp.csv', './Datasets/N_BaIoT/4.mirai.ack.csv', './Datasets/N_BaIoT/4.mirai.scan.csv', './Datasets/N_BaIoT/4.mirai.syn.csv', './Datasets/N_BaIoT/4.mirai.udp.csv', './Datasets/N_BaIoT/4.mirai.udpplain.csv'), ignore_index = True)
datatrain = ReadDataFromCSV('./Datasets/N_BaIoT/5.benign.csv', './Datasets/N_BaIoT/5.gafgyt.combo.csv', './Datasets/N_BaIoT/5.gafgyt.junk.csv', './Datasets/N_BaIoT/5.gafgyt.scan.csv', './Datasets/N_BaIoT/5.gafgyt.udp.csv', './Datasets/N_BaIoT/5.mirai.ack.csv', './Datasets/N_BaIoT/5.mirai.scan.csv', './Datasets/N_BaIoT/5.mirai.syn.csv', './Datasets/N_BaIoT/5.mirai.udp.csv', './Datasets/N_BaIoT/5.mirai.udpplain.csv')
datatest = ReadDataFromCSV('./Datasets/N_BaIoT/2.benign.csv', './Datasets/N_BaIoT/2.gafgyt.combo.csv', './Datasets/N_BaIoT/2.gafgyt.junk.csv', './Datasets/N_BaIoT/2.gafgyt.scan.csv', './Datasets/N_BaIoT/2.gafgyt.udp.csv', './Datasets/N_BaIoT/2.mirai.ack.csv', './Datasets/N_BaIoT/2.mirai.scan.csv', './Datasets/N_BaIoT/2.mirai.syn.csv', './Datasets/N_BaIoT/2.mirai.udp.csv', './Datasets/N_BaIoT/2.mirai.udpplain.csv', isTest=True)
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

_, _, train_data_loader = DataPreprocessing(datatrain, 500000, BATCH_SIZE)
X_test, y_test, test_data_loader = DataPreprocessing(datatest, 500000, 1000)
n_feature = X_test.shape[1]
n_class = np.unique(y_test).shape[0]

print("Number of testing features : ", n_feature)
print("Number of testing classes : ", n_class)

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def train(model, train_data_loader, test_data_loader, epoch):
    optimizer = optim.Adam(model.parameters(), lr=lr)

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

    return model

def test(model, test_data_loader, epoch=None):
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
    acc = 100. * correct / total
    if epoch:
        print('Test set epoch {}: Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            int(epoch), loss.item(), correct, total, acc))  
    
    return acc

model = None

worker = Worker()
worker.create_socket()
worker.connect()
worker.recv_train_model()
