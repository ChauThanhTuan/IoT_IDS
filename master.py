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

mean_standard = {'MI_dir_L5_weight': 93.26944471807772, 'MI_dir_L5_mean': 191.4107472827913, 'MI_dir_L5_variance': 18082.36561902601, 'MI_dir_L3_weight': 147.79771305906476, 'MI_dir_L3_mean': 191.78709449963588, 'MI_dir_L3_variance': 20115.809813276137, 'MI_dir_L1_weight': 422.69230621404853, 'MI_dir_L1_mean': 192.45398849572408, 'MI_dir_L1_variance': 21632.825106993005, 'MI_dir_L0.1_weight': 3956.491492914809, 'MI_dir_L0.1_mean': 193.0699092738356, 'MI_dir_L0.1_variance': 22237.066016856134, 'MI_dir_L0.01_weight': 23793.412747541897, 'MI_dir_L0.01_mean': 193.01654565510844, 'MI_dir_L0.01_variance': 22539.9332115765, 'H_L5_weight': 93.26944581738746, 'H_L5_mean': 191.41075164554763, 'H_L5_variance': 18082.366468211738, 'H_L3_weight': 147.79771466815876, 'H_L3_mean': 191.78710590340305, 'H_L3_variance': 20115.81218825291, 'H_L1_weight': 422.6923093676522, 'H_L1_mean': 192.45402679467415, 'H_L1_variance': 21632.832355214105, 'H_L0.1_weight': 3956.491520871961, 'H_L0.1_mean': 193.06988798218413, 'H_L0.1_variance': 22237.128611485674, 'H_L0.01_weight': 23793.41297913415, 'H_L0.01_mean': 193.01641747208495, 'H_L0.01_variance': 22540.042839561123, 'HH_L5_weight': 48.57348305538757, 'HH_L5_mean': 192.97834430973938, 'HH_L5_std': 1.7385065758928482, 'HH_L5_magnitude': 202.13845242297634, 'HH_L5_radius': 1015.1592102813747, 'HH_L5_covariance': -42.26661944608412, 'HH_L5_pcc': -0.00017076370448230644, 'HH_L3_weight': 78.8458688800357, 'HH_L3_mean': 192.94886524266582, 'HH_L3_std': 1.8975861392031632, 'HH_L3_magnitude': 202.14147930963802, 'HH_L3_radius': 1076.251172271592, 'HH_L3_covariance': -54.94590016968729, 'HH_L3_pcc': 7.17244348425036e-06, 'HH_L1_weight': 228.81715364268152, 'HH_L1_mean': 192.88272783991468, 'HH_L1_std': 2.4522577940438013, 'HH_L1_magnitude': 202.0854965121485, 'HH_L1_radius': 1157.949955032346, 'HH_L1_covariance': -71.42757321442832, 'HH_L1_pcc': 0.0007375210638640153, 'HH_L0.1_weight': 2080.7310406959023, 'HH_L0.1_mean': 192.91933177259202, 'HH_L0.1_std': 4.307469206483778, 'HH_L0.1_magnitude': 201.85908207058426, 'HH_L0.1_radius': 1455.4437111175168, 'HH_L0.1_covariance': -55.78491325953138, 'HH_L0.1_pcc': 0.006694613707242562, 'HH_L0.01_weight': 11026.991675978004, 'HH_L0.01_mean': 193.07143610205424, 'HH_L0.01_std': 5.520802243195934, 'HH_L0.01_magnitude': 201.941289227371, 'HH_L0.01_radius': 1800.2646738286505, 'HH_L0.01_covariance': 24.097176851819402, 'HH_L0.01_pcc': 0.012069709509734518, 'HH_jit_L5_weight': 48.57348305538757, 'HH_jit_L5_mean': 628687983.5799067, 'HH_jit_L5_variance': 112340819440481.78, 'HH_jit_L3_weight': 78.8458688800357, 'HH_jit_L3_mean': 628751145.8463289, 'HH_jit_L3_variance': 197641244861511.25, 'HH_jit_L1_weight': 228.8171536426819, 'HH_jit_L1_mean': 631157099.7313951, 'HH_jit_L1_variance': 3405251850774851.5, 'HH_jit_L0.1_weight': 2080.7310406959123, 'HH_jit_L0.1_mean': 643901119.4025023, 'HH_jit_L0.1_variance': 1.4570148806055658e+16, 'HH_jit_L0.01_weight': 11026.99167597793, 'HH_jit_L0.01_mean': 647138980.8789642, 'HH_jit_L0.01_variance': 1.6353972281457586e+16, 'HpHp_L5_weight': 4.155876790085, 'HpHp_L5_mean': 193.04962834021617, 'HpHp_L5_std': 1.0465706575651, 'HpHp_L5_magnitude': 200.1744897264621, 'HpHp_L5_radius': 599.9943653451037, 'HpHp_L5_covariance': 52.23516013649219, 'HpHp_L5_pcc': 0.0009395902603454411, 'HpHp_L3_weight': 6.114202414667558, 'HpHp_L3_mean': 193.03115333530116, 'HpHp_L3_std': 1.0882708603848092, 'HpHp_L3_magnitude': 200.17632192361475, 'HpHp_L3_radius': 652.1704052924346, 'HpHp_L3_covariance': 54.545628664337436, 'HpHp_L3_pcc': 0.001024547713182817, 'HpHp_L1_weight': 15.799919287955433, 'HpHp_L1_mean': 192.97496642227657, 'HpHp_L1_std': 1.1836082766974791, 'HpHp_L1_magnitude': 200.13403557831117, 'HpHp_L1_radius': 711.8888868541, 'HpHp_L1_covariance': 57.98777479891492, 'HpHp_L1_pcc': 0.0008930839307075592, 'HpHp_L0.1_weight': 142.17035985035372, 'HpHp_L0.1_mean': 192.87213318872682, 'HpHp_L0.1_std': 1.6271694757690296, 'HpHp_L0.1_magnitude': 199.81857577779715, 'HpHp_L0.1_radius': 942.5901007011578, 'HpHp_L0.1_covariance': 63.37947466920262, 'HpHp_L0.1_pcc': 0.0015511964391728128, 'HpHp_L0.01_weight': 809.4731319852671, 'HpHp_L0.01_mean': 192.76908789601808, 'HpHp_L0.01_std': 1.820125342690955, 'HpHp_L0.01_magnitude': 199.79717965555466, 'HpHp_L0.01_radius': 1261.4587873489922, 'HpHp_L0.01_covariance': 77.5912223343027}
std_standard = {'MI_dir_L5_weight': 65.40761114627169, 'MI_dir_L5_mean': 167.18636142867794, 'MI_dir_L5_variance': 25923.318386269715, 'MI_dir_L3_weight': 103.93752816203764, 'MI_dir_L3_mean': 161.14941602593373, 'MI_dir_L3_variance': 27186.26475124098, 'MI_dir_L1_weight': 300.74232681428316, 'MI_dir_L1_mean': 156.6659727830628, 'MI_dir_L1_variance': 28037.96437162659, 'MI_dir_L0.1_weight': 2830.405540052882, 'MI_dir_L0.1_mean': 155.26547351863826, 'MI_dir_L0.1_variance': 28313.83798639461, 'MI_dir_L0.01_weight': 20218.425533662754, 'MI_dir_L0.01_mean': 155.04640981959292, 'MI_dir_L0.01_variance': 27978.396451801727, 'H_L5_weight': 65.40760961166421, 'H_L5_mean': 167.18636536256307, 'H_L5_variance': 25923.317819656284, 'H_L3_weight': 103.9375259101526, 'H_L3_mean': 161.1494271222162, 'H_L3_variance': 27186.263166109795, 'H_L1_weight': 300.74232241340354, 'H_L1_mean': 156.66601331155354, 'H_L1_variance': 28037.959803860016, 'H_L0.1_weight': 2830.405500995867, 'H_L0.1_mean': 155.26513358818116, 'H_L0.1_variance': 28313.801818043023, 'H_L0.01_weight': 20218.42526116254, 'H_L0.01_mean': 155.04558179145764, 'H_L0.01_variance': 27978.330330981662, 'HH_L5_weight': 59.85275070845524, 'HH_L5_mean': 214.460293588465, 'HH_L5_std': 19.761329690953502, 'HH_L5_magnitude': 215.32578140500848, 'HH_L5_radius': 16874.37846281378, 'HH_L5_covariance': 2341.251085216031, 'HH_L5_pcc': 0.03167116955031993, 'HH_L3_weight': 97.38626974617111, 'HH_L3_mean': 214.3546851629615, 'HH_L3_std': 19.94646451124956, 'HH_L3_magnitude': 215.2227234995502, 'HH_L3_radius': 17612.773354575864, 'HH_L3_covariance': 2591.317219594441, 'HH_L3_pcc': 0.03675175653358045, 'HH_L1_weight': 283.61450698767186, 'HH_L1_mean': 214.21400221355728, 'HH_L1_std': 20.20883373338824, 'HH_L1_magnitude': 215.06417214457423, 'HH_L1_radius': 18252.492709663617, 'HH_L1_covariance': 2851.96284197137, 'HH_L1_pcc': 0.049585684789514924, 'HH_L0.1_weight': 2531.2952547805357, 'HH_L0.1_mean': 214.09245572224572, 'HH_L0.1_std': 22.700448615886838, 'HH_L0.1_magnitude': 214.38883333782934, 'HH_L0.1_radius': 19198.99644085447, 'HH_L0.1_covariance': 2773.986437822168, 'HH_L0.1_pcc': 0.08055039304760538, 'HH_L0.01_weight': 14382.975460265521, 'HH_L0.01_mean': 214.1395784684696, 'HH_L0.01_std': 25.348287398534648, 'HH_L0.01_magnitude': 214.0047239780351, 'HH_L0.01_radius': 19339.88973720824, 'HH_L0.01_covariance': 2253.7581059384424, 'HH_L0.01_pcc': 0.09580042702307627, 'HH_jit_L5_weight': 59.85275070845524, 'HH_jit_L5_mean': 743016473.5743761, 'HH_jit_L5_variance': 6250734702901485.0, 'HH_jit_L3_weight': 97.38626974617115, 'HH_jit_L3_mean': 742969733.7671242, 'HH_jit_L3_variance': 6950980850854370.0, 'HH_jit_L1_weight': 283.61450698767254, 'HH_jit_L1_mean': 741207524.8975542, 'HH_jit_L1_variance': 2.7593568000854604e+16, 'HH_jit_L0.1_weight': 2531.295254780552, 'HH_jit_L0.1_mean': 735639959.680993, 'HH_jit_L0.1_variance': 8.302410347809195e+16, 'HH_jit_L0.01_weight': 14382.975460265423, 'HH_jit_L0.01_mean': 734900370.1353989, 'HH_jit_L0.01_variance': 9.062917156568589e+16, 'HpHp_L5_weight': 14.916354189488633, 'HpHp_L5_mean': 214.90560246281223, 'HpHp_L5_std': 16.31536442623088, 'HpHp_L5_magnitude': 216.5048671747931, 'HpHp_L5_radius': 11731.921287501496, 'HpHp_L5_covariance': 1355.550615852965, 'HpHp_L5_pcc': 0.031559148213155364, 'HpHp_L3_weight': 23.813137037069808, 'HpHp_L3_mean': 214.77739782553834, 'HpHp_L3_std': 16.977476241343275, 'HpHp_L3_magnitude': 216.40067325338478, 'HpHp_L3_radius': 12738.551333658606, 'HpHp_L3_covariance': 1441.8509375716196, 'HpHp_L3_pcc': 0.03536669461717413, 'HpHp_L1_weight': 67.87674065863807, 'HpHp_L1_mean': 214.5814633265508, 'HpHp_L1_std': 17.771723529853336, 'HpHp_L1_magnitude': 216.23883366182812, 'HpHp_L1_radius': 13903.015496614113, 'HpHp_L1_covariance': 1596.845577880396, 'HpHp_L1_pcc': 0.04371894878750802, 'HpHp_L0.1_weight': 641.8053849642735, 'HpHp_L0.1_mean': 214.34376453013917, 'HpHp_L0.1_std': 18.8604528320339, 'HpHp_L0.1_magnitude': 215.67686798817374, 'HpHp_L0.1_radius': 15893.714854713837, 'HpHp_L0.1_covariance': 1878.607200061036, 'HpHp_L0.1_pcc': 0.05649817929768701, 'HpHp_L0.01_weight': 3880.2362089838475, 'HpHp_L0.01_mean': 214.1068456947141, 'HpHp_L0.01_std': 18.309516854040915, 'HpHp_L0.01_magnitude': 215.2659136261106, 'HpHp_L0.01_radius': 17747.069353764848, 'HpHp_L0.01_covariance': 2453.3182055179027}


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

                    
                    LZ4_Start = time.time()
                    LZ4_DATA = LZ4.compress(response)
                    LZ4_End = time.time()
                    # SNAPPY_DATA = SNAPPY.compress(response)
                    BZ2_DATA = BZ2.compress(response)
                    BZ2_End = time.time()
                    LZMA_DATA = LZMA.compress(response)
                    LZMA_End = time.time()
                    GZIP_DATA = GZIP.compress(response)
                    GZIP_End = time.time()

                    print("===================================Compress======================================")
                    print("Data Size:")
                    print("  Input:  %d" %          len(response))
                    print("  LZ4:    %d (%.2f) - (%.2f)s" %   (len(LZ4_DATA), len(LZ4_DATA) / float(len(response)), LZ4_End - LZ4_Start))
                    # print("  Snappy: %d (%.2f) - (%.2f)s" %   (len(SNAPPY_DATA), len(SNAPPY_DATA) / float(len(response)), GZIP_End - LZMA_End))
                    print("  BZ2:    %d (%.2f) - (%.2f)s" %   (len(BZ2_DATA), len(BZ2_DATA) / float(len(response)), BZ2_End - LZ4_End))
                    print("  LZMA:   %d (%.2f) - (%.2f)s" %   (len(LZMA_DATA), len(LZMA_DATA) / float(len(response)), LZMA_End - BZ2_End))
                    print("  GZIP:   %d (%.2f) - (%.2f)s" %   (len(GZIP_DATA), len(GZIP_DATA) / float(len(response)), GZIP_End - LZMA_End))
                    print("=================================================================================")


                    LZ4_Start = time.time()
                    LZ4_DECOMPRESS = LZ4.decompress(LZ4_DATA)
                    LZ4_End = time.time()
                    BZ2_DECOMPRESS = BZ2.decompress(BZ2_DATA)
                    BZ2_End = time.time()
                    LZMA_DECOMPRESS = LZMA.decompress(LZMA_DATA)
                    LZMA_End = time.time()
                    GZIP_DECOMPRESS = GZIP.decompress(GZIP_DATA)
                    GZIP_End = time.time()

                    print("===================================Decompress====================================")
                    print("Data Size:")
                    print("  LZ4:    %d (%.2f) - (%.2f)s" %   (len(LZ4_DECOMPRESS), len(LZ4_DECOMPRESS) / float(len(response)), LZ4_End - LZ4_Start))
                    # print("  Snappy: %d (%.2f) - (%.2f)s" %   (len(SNAPPY_DECOMPRESS), len(SNAPPY_DECOMPRESS) / float(len(response)), GZIP_End - LZMA_End))
                    print("  BZ2:    %d (%.2f) - (%.2f)s" %   (len(BZ2_DECOMPRESS), len(BZ2_DECOMPRESS) / float(len(response)), BZ2_End - LZ4_End))
                    print("  LZMA:   %d (%.2f) - (%.2f)s" %   (len(LZMA_DECOMPRESS), len(LZMA_DECOMPRESS) / float(len(response)), LZMA_End - BZ2_End))
                    print("  GZIP:   %d (%.2f) - (%.2f)s" %   (len(GZIP_DECOMPRESS), len(GZIP_DECOMPRESS) / float(len(response)), GZIP_End - LZMA_End))
                    print("=================================================================================")

                    print("Replying to the Client.")
                    self.connection.sendall(GZIP_DATA)
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
        torch.save(model.state_dict(), "IoT_Intrusions_Detection.pth")

        # Reload the model in a new model object
        model_new = Net(n_feature,n_class)
        model_new.load_state_dict(torch.load("IoT_Intrusions_Detection.pth"))
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
data = data[:700000]

threat_types = data["type"].values
encoder = LabelEncoder()
# use LabelEncoder to encode the threat types in numeric values
y_test = encoder.fit_transform(threat_types)
print("Shape of target vector : ", y_test.shape)

#drop labels from training dataset
data=data.drop(columns='type')

#standardize numerical columns
def standardize(df,col):
    df[col]= (df[col]-mean_standard[col])/std_standard[col]

for i in (data.iloc[:,:-1].columns):
    standardize(data,i)

X_test = data.values
del data

n_feature = X_test.shape[1]
n_class = np.unique(y_test).shape[0]

print("Number of testing features : ", n_feature)
print("Number of testing classes : ", n_class)

BATCH_SIZE = 700

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
