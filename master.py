import socket
import pickle
import threading
from threading import Thread
import time
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

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from compress import Compressor

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
    
class Master():
    
    def __init__(self):
        super().__init__()

    def create_socket(self, *args):
        self.soc = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)

    def bind_socket(self, *args):
        ip = "0.0.0.0"
        port = "22334"
        self.soc.bind((ip, int(port)))
        print(f"Socket Bound to {ip}:{port}")

    def listen_accept(self, *args):
        self.soc.listen(1)
        print("Socket is Listening for Connections")

        self.listenThread = ListenThread(master=self)
        self.listenThread.start()

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

class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, master, buffer_size=1024, recv_timeout=5):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout
        self.master = master

    def recv(self):
        all_data_received_flag = False
        received_data = b""
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data += data

                try:
                    pickle.loads(received_data)
                    # If the previous pickle.loads() statement is passed, this means all the data is received.
                    # Thus, no need to continue the loop. The flag all_data_received_flag is set to True to signal all data is received.
                    all_data_received_flag = True
                except BaseException:
                    # An exception is expected when the data is not 100% received.
                    pass

                if data == b'': # Nothing received from the client.
                    received_data = b""
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute, return with status 0 to close the connection.
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0 # 0 means the connection is no longer active and it should be closed.

                elif all_data_received_flag:
                    print(f"All data ({len(received_data)} bytes) Received from {self.client_info}.")

                    if len(received_data) > 0:
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1

                        except BaseException as e:
                            print(f"Error Decoding the Client's Data: {e}.\n")
                            return None, 0

                else:
                    # In case data are received from the client, update the recv_start_time to the current time to reset the timeout counter.
                    self.recv_start_time = time.time()

            except BaseException as e:
                print(f"Error Receiving Data from the Client: {e}.\n")
                return None, 0


    def reply(self, received_data):
        global model, epoch, local_models, num_connection, workers, average_models, HAS_TESTED
        response = None
        
        if (type(received_data) is dict):
            if (("data" in received_data.keys()) and ("subject" in received_data.keys())):
                subject = received_data["subject"]
                # print(f"Client's Message Subject is {subject}.")

                if subject == "echo":
                    data = {"subject": "model", "data": model}
                    
                    try:
                        response = pickle.dumps(data)
                    except BaseException as e:
                        print("Error Encoding the Message: {e}.\n")
                elif subject == "model":
                    try:
                        model_received = received_data["data"]
                        local_models.append(model_received)

                        epoch += 1/num_connection
                        HAS_TESTED = False

                        if epoch.is_integer():
                            # create a thread for the function test
                            average_thread = ThreadWithReturnValue(target=AverageModel, args=([local_models]))
                            # start the thread
                            average_thread.start()
                            # wait for the thread to finish
                            average_model = average_thread.join()
                            
                            if average_model != None:
                                average_models.append(average_model)
                            else:
                                average_models.append(average_models[int(epoch) - 2])
                            local_models = []

                            # create a thread for the function test
                            test_thread = Thread(target=test, args=(average_models[int(epoch) - 1], data_loader))
                            # start the thread
                            test_thread.start()
                            # wait for the thread to finish
                            test_thread.join()

                            HAS_TESTED = True
                        else:
                            while HAS_TESTED == False:
                                time.sleep(1)
                        
                        data = {"subject": "model", "data": average_models[int(epoch) - 1]}
                        response = pickle.dumps(data)

                    except BaseException as e:
                        print(f"reply(): Error Decoding the Client's Data: {e}.\n")
                else:
                    num_connection -= 1
                    del workers[self.client_info]
                    return
                    # response = pickle.dumps("Response from the master")
                            
                try:
                    # print(self.connection)

                    LZ4_DATA = LZ4.compress(response)
                    # SNAPPY_DATA = SNAPPY.compress(response)
                    BZ2_DATA = BZ2.compress(response)
                    LZMA_DATA = LZMA.compress(response)
                    GZIP_DATA = GZIP.compress(response)

                    print("=================================================================================")
                    print("Data Size:")
                    print("  Input:  %d" %          len(response))
                    print("  LZ4:    %d (%.2f)" %   (len(LZ4_DATA), len(LZ4_DATA) / float(len(response))))
                    # print("  Snappy: %d (%.2f)" %   (len(SNAPPY_DATA), len(SNAPPY_DATA) / float(len(response))))
                    print("  BZ2:    %d (%.2f)" %   (len(BZ2_DATA), len(BZ2_DATA) / float(len(response))))
                    print("  LZMA:   %d (%.2f)" %   (len(LZMA_DATA), len(LZMA_DATA) / float(len(response))))
                    print("  GZIP:   %d (%.2f)" %   (len(GZIP_DATA), len(GZIP_DATA) / float(len(response))))
                    print("=================================================================================")

                    print("Replying to the Client.")
                    self.connection.sendall(response)
                except BaseException as e:
                    print(f"Error Sending Data to the Client: {e}.\n")

            else:
                print(f"The received dictionary from the client must have the 'subject' and 'data' keys available. The existing keys are {received_data.keys()}.")
                print("Error Parsing Received Dictionary")
        else:
            print(f"A dictionary is expected to be received from the client but {type(received_data)} received.")

    def run(self):
        print(f"Running a Thread for the Connection with {self.client_info}.")

        # This while loop allows the master to wait for the client to send data more than once within the same connection.
        while len(workers) == NUM_WORKERS:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = f"Waiting to Receive Data Starting from {time_struct.tm_mday}/{time_struct.tm_mon}/{time_struct.tm_year} {time_struct.tm_hour}:{time_struct.tm_min}:{time_struct.tm_sec} GMT"
            print(date_time)

            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print(f"Connection Closed with {self.client_info} either due to inactivity for {self.recv_timeout} seconds or due to an error.\n")
                break
            
            # print(received_data)
            self.reply(received_data)
        
        print("Model is Trained")
        master.close_socket()

        # Save the model
        torch.save(model.state_dict(), "IoT_Intrusions_Detection.pt")

        # Reload the model in a new model object
        model_new = Net(n_feature,n_class)
        model_new.load_state_dict(torch.load("IoT_Intrusions_Detection.pt"))
        model_new.eval()
        
        y_true = test_labels
        y_pred = []

        for item in test_inputs:
            item = item.reshape(1, n_feature, 1)
            pred = model_new(item)
            pred = int(pred.argmax().data.cpu().numpy())
            y_pred.append(pred)
        target_names = ['benign', 'gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_udp', 'mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udp', 'mirai_udpplain']
        print(classification_report(y_true, y_pred, target_names=target_names))

class ListenThread(threading.Thread):

    def __init__(self, master):
        threading.Thread.__init__(self)
        self.master = master

    def run(self):
        global num_connection, workers
        while num_connection < NUM_WORKERS:
            try:
                connection, client_info = self.master.soc.accept()
                num_connection += 1
                print(f"New Connection from {client_info}")
                print("Number of connection: ", num_connection)

                socket_thread = SocketThread(connection=connection,
                                             client_info=client_info, 
                                             master=self.master,
                                             buffer_size=1024,
                                             recv_timeout=3600)
                workers[client_info] = socket_thread
                
            except BaseException as e:
                self.master.soc.close()
                print(f"Error in the run() of the ListenThread class: {e}.\n")
                print("Socket is No Longer Accepting Connections")
                break  
        
        for worker in workers:
            workers[worker].start()

def ReadDataFromCSV(benign, g_c, g_j, g_s, g_u, m_a, m_sc, m_sy, m_u, m_u_p):

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

# data = ReadDataFromCSV('./Datasets/N_BaIoT/1.benign.csv', './Datasets/N_BaIoT/1.gafgyt.combo.csv', './Datasets/N_BaIoT/1.gafgyt.junk.csv', './Datasets/N_BaIoT/1.gafgyt.scan.csv', './Datasets/N_BaIoT/1.gafgyt.udp.csv', './Datasets/N_BaIoT/1.mirai.ack.csv', './Datasets/N_BaIoT/1.mirai.scan.csv', './Datasets/N_BaIoT/1.mirai.syn.csv', './Datasets/N_BaIoT/1.mirai.udp.csv', './Datasets/N_BaIoT/1.mirai.udpplain.csv')
data = ReadDataFromCSV('./Datasets/N_BaIoT/2.benign.csv', './Datasets/N_BaIoT/2.gafgyt.combo.csv', './Datasets/N_BaIoT/2.gafgyt.junk.csv', './Datasets/N_BaIoT/2.gafgyt.scan.csv', './Datasets/N_BaIoT/2.gafgyt.udp.csv', './Datasets/N_BaIoT/2.mirai.ack.csv', './Datasets/N_BaIoT/2.mirai.scan.csv', './Datasets/N_BaIoT/2.mirai.syn.csv', './Datasets/N_BaIoT/2.mirai.udp.csv', './Datasets/N_BaIoT/2.mirai.udpplain.csv')
# data = data.append(ReadDataFromCSV('./Datasets/N_BaIoT/2.benign.csv', './Datasets/N_BaIoT/2.gafgyt.combo.csv', './Datasets/N_BaIoT/2.gafgyt.junk.csv', './Datasets/N_BaIoT/2.gafgyt.scan.csv', './Datasets/N_BaIoT/2.gafgyt.udp.csv', './Datasets/N_BaIoT/2.mirai.ack.csv', './Datasets/N_BaIoT/2.mirai.scan.csv', './Datasets/N_BaIoT/2.mirai.syn.csv', './Datasets/N_BaIoT/2.mirai.udp.csv', './Datasets/N_BaIoT/2.mirai.udpplain.csv'), ignore_index = True)
# data = data.append(ReadDataFromCSV('./Datasets/N_BaIoT/4.benign.csv', './Datasets/N_BaIoT/4.gafgyt.combo.csv', './Datasets/N_BaIoT/4.gafgyt.junk.csv', './Datasets/N_BaIoT/4.gafgyt.scan.csv', './Datasets/N_BaIoT/4.gafgyt.udp.csv', './Datasets/N_BaIoT/4.mirai.ack.csv', './Datasets/N_BaIoT/4.mirai.scan.csv', './Datasets/N_BaIoT/4.mirai.syn.csv', './Datasets/N_BaIoT/4.mirai.udp.csv', './Datasets/N_BaIoT/4.mirai.udpplain.csv'), ignore_index = True)
# data = data.append(ReadDataFromCSV('./Datasets/N_BaIoT/5.benign.csv', './Datasets/N_BaIoT/5.gafgyt.combo.csv', './Datasets/N_BaIoT/5.gafgyt.junk.csv', './Datasets/N_BaIoT/5.gafgyt.scan.csv', './Datasets/N_BaIoT/5.gafgyt.udp.csv', './Datasets/N_BaIoT/5.mirai.ack.csv', './Datasets/N_BaIoT/5.mirai.scan.csv', './Datasets/N_BaIoT/5.mirai.syn.csv', './Datasets/N_BaIoT/5.mirai.udp.csv', './Datasets/N_BaIoT/5.mirai.udpplain.csv'), ignore_index = True)
# data = data.append(ReadDataFromCSV('./Datasets/N_BaIoT/8.benign.csv', './Datasets/N_BaIoT/8.gafgyt.combo.csv', './Datasets/N_BaIoT/8.gafgyt.junk.csv', './Datasets/N_BaIoT/8.gafgyt.scan.csv', './Datasets/N_BaIoT/8.gafgyt.udp.csv', './Datasets/N_BaIoT/8.mirai.ack.csv', './Datasets/N_BaIoT/8.mirai.scan.csv', './Datasets/N_BaIoT/8.mirai.syn.csv', './Datasets/N_BaIoT/8.mirai.udp.csv', './Datasets/N_BaIoT/8.mirai.udpplain.csv'), ignore_index = True)
# data = data.append(ReadDataFromCSV('./Datasets/N_BaIoT/9.benign.csv', './Datasets/N_BaIoT/9.gafgyt.combo.csv', './Datasets/N_BaIoT/9.gafgyt.junk.csv', './Datasets/N_BaIoT/9.gafgyt.scan.csv', './Datasets/N_BaIoT/9.gafgyt.udp.csv', './Datasets/N_BaIoT/9.mirai.ack.csv', './Datasets/N_BaIoT/9.mirai.scan.csv', './Datasets/N_BaIoT/9.mirai.syn.csv', './Datasets/N_BaIoT/9.mirai.udp.csv', './Datasets/N_BaIoT/9.mirai.udpplain.csv'), ignore_index = True)

#how many instances of each class
data.groupby('type')['type'].count()

#shuffle rows of dataframe 
sampler=np.random.permutation(len(data))
data=data.take(sampler)
data = data[:500000]

threat_types = data["type"].values
encoder = LabelEncoder()
# use LabelEncoder to encode the threat types in numeric values
y_test = encoder.fit_transform(threat_types)
print("Shape of target vector : ", y_test.shape)

#drop labels from training dataset
data=data.drop(columns='type')

#standardize numerical columns
def standardize(df,col):
    df[col]= (df[col]-df[col].mean())/df[col].std()

data_st=data.copy()
for i in (data_st.iloc[:,:-1].columns):
    standardize(data_st,i)

X_test = data_st.values


n_feature = X_test.shape[1]
n_class = np.unique(y_test).shape[0]

print("Number of testing features : ", n_feature)
print("Number of testing classes : ", n_class)

BATCH_SIZE = 1000

# Create pytorch tensor from X_test, y_test
test_inputs = torch.tensor(X_test,dtype=torch.float)
test_labels = torch.tensor(y_test).type(torch.LongTensor)

dataset = TensorDataset(test_inputs, test_labels)
data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

def AverageModel(local_models):
    local_models = [model for model in local_models if model is not None]

    num_model = len(local_models)
    print("num_model: ", num_model)
    if num_model == 0:
        return None
    if num_model == 1:
        return local_models[0]

    average_models = local_models[0]
    local_models.remove(average_models)

    average_models_state = dict(average_models.state_dict().items())

    for model in local_models:
        model_state = dict(model.state_dict().items())

        for state in average_models.state_dict():
            average_models_state[state] += model_state[state]

    for state in average_models.state_dict():
        average_models_state[state] /= num_model
    
    return average_models

def test(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    for (data, target) in data_loader:
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
    
    acc = 100. * correct / total
    print('Test set epoch {}: Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        int(epoch), loss.item(), correct, total, acc))    
    
NUM_WORKERS = 2
HAS_TESTED = False
model = Net(n_feature, n_class)    
epoch = 0
num_connection = 0
workers = {}
local_models = []
average_models = []

LZ4 = Compressor()
# SNAPPY = Compressor()
BZ2 = Compressor()
LZMA = Compressor()
GZIP = Compressor()

LZ4.use_lz4()
# SNAPPY.use_snappy()
BZ2.use_bz2()
LZMA.use_lzma()
GZIP.use_gzip()

test(model, data_loader)

master = Master()
master.create_socket()
master.bind_socket()
master.listen_accept()
