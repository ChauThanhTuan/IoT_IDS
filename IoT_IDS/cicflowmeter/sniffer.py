import argparse

from scapy.sendrecv import AsyncSniffer

from flow_session import generate_session_class

import streamlit as st


def create_sniffer(
    input_file, input_interface, output_mode, output_file, url_model=None
):
    assert (input_file is None) ^ (input_interface is None)

    NewFlowSession = generate_session_class(output_mode, output_file, url_model)

    if input_file is not None:
        return AsyncSniffer(
            offline=input_file,
            filter="ip and (tcp or udp)",
            prn=None,
            session=NewFlowSession,
            store=False,
        )
    else:
        return AsyncSniffer(
            iface=input_interface,
            filter="ip and (tcp or udp)",
            prn=None,
            session=NewFlowSession,
            store=False,
        )


def main():
    # parser = argparse.ArgumentParser()

    # input_group = parser.add_mutually_exclusive_group(required=True)
    # input_group.add_argument(
    #     "-i",
    #     "--interface",
    #     action="store",
    #     dest="input_interface",
    #     help="capture online data from INPUT_INTERFACE",
    # )

    # input_group.add_argument(
    #     "-f",
    #     "--file",
    #     action="store",
    #     dest="input_file",
    #     help="capture offline data from INPUT_FILE",
    # )

    # output_group = parser.add_mutually_exclusive_group(required=False)
    # output_group.add_argument(
    #     "-c",
    #     "--csv",
    #     "--flow",
    #     action="store_const",
    #     const="flow",
    #     dest="output_mode",
    #     help="output flows as csv",
    # )

    # url_model = parser.add_mutually_exclusive_group(required=False)
    # url_model.add_argument(
    #     "-u",
    #     "--url",
    #     action="store",
    #     dest="url_model",
    #     help="URL endpoint for send to Machine Learning Model. e.g http://0.0.0.0:80/prediction",
    # )

    # parser.add_argument(
    #     "output",
    #     help="output file name (in flow mode) or directory (in sequence mode)",
    # )

    # args = parser.parse_args()

    # sniffer = create_sniffer(
    #     args.input_file,
    #     args.input_interface,
    #     args.output_mode,
    #     args.output,
    #     args.url_model,
    # )
    sniffer = create_sniffer(None, "Wi-Fi", "--flow", "output", None)
    st.runtime.scriptrunner.script_run_context.get_script_run_ctx(sniffer)
    sniffer.start()

    try:
        sniffer.join()
    except KeyboardInterrupt:
        sniffer.stop()
    finally:
        sniffer.join()


import logging
import os, time 
import sys
import numpy as np 
import requests
import threading
from threading import Thread
import streamlit
st_ver = int(streamlit.__version__.replace('.',''))
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

from load_css import local_css

local_css("style.css")

from io import BytesIO
from glob import glob
from PIL import Image, ImageEnhance

# import openai
import pandas as pd
import streamlit as st
# import io


import subprocess
import subprocess as sp

import sys
# from  master import workers 


import torch 
# import the necessary packages
from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import MaxPool1d
from torch.nn import Dropout
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import BatchNorm1d
from torch import flatten

import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from functionforDownloadButtons import download_button


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
    


sys.path.insert(0, ".")



gallery_files = glob(os.path.join(".", "images", "*"))
gallery_dict = {image_path.split("/")[-1].split(".")[-2].replace("-", " "): image_path
    for image_path in gallery_files}

st.image("./images/fl.png")
st.sidebar.markdown("## NetWork Threat Prediction System Using Federated Learning 🎨")
# st.markdown("## NetWork Threat Prediction System Using Federated Learning 🎨")

with st.sidebar.expander("ℹ️ - About this app", expanded=True):
    st.sidebar.caption(
        """     
 This system allows you to upload a CSV. It uses trained model using Federated learning to assist you in predicting your questions about the network traffic, which is normal or threat.
	   """
    )

# st.sidebar.caption("Keep your data private.")
st.sidebar.markdown("Made by [Chau Thanh Tuan](https://www.linkedin.com/in/siavash-yasini/) , [Nguyen My Quynh](https://www.linkedin.com/in/siavash-yasini/)")

st.sidebar.write("---\n")
st.sidebar.caption("""You can check out the source code [here](https://github.com/syasini/sophisticated_palette).
                      """)





# toggle = st.sidebar.checkbox("Toggle Update", value=True, help="Continuously update the pallete with every change in the app.")
# click = st.sidebar.button("Find Palette", disabled=bool(toggle))

# st.sidebar.markdown("---")
# st.sidebar.header("Settings")
# palette_size = int(st.sidebar.number_input("palette size", min_value=1, max_value=20, value=5, step=1, help="Number of colors to infer from the image."))
# sample_size = int(st.sidebar.number_input("sample size", min_value=5, max_value=3000, value=500, step=500, help="Number of sample pixels to pick from the image."))


# Image Enhancement
# enhancement_categories = enhancement_range.keys()
# enh_expander = st.sidebar.expander("Image Enhancements", expanded=False)
# with enh_expander:
    
#     if st.button("reset"):
#         for cat in enhancement_categories:
#             if f"{cat}_enhancement" in st.session_state:
#                 st.session_state[f"{cat}_enhancement"] = 1.0
# enhancement_factor_dict = {
#     cat: enh_expander.slider(f"{cat} Enhancement", 
#                             value=1., 
#                             min_value=enhancement_range[cat][0], 
#                             max_value=enhancement_range[cat][1], 
#                             step=enhancement_range[cat][2],
#                             key=f"{cat}_enhancement")
#     for cat in enhancement_categories
# }
# enh_expander.info("**Try the following**\n\nColor Enhancements = 2.6\n\nContrast Enhancements = 1.1\n\nBrightness Enhancements = 1.1")

# # Clustering Model 
# model_name = st.sidebar.selectbox("machine learning model", model_dict.keys(), help="Machine Learning model to use for clustering pixels and colors together.")
# sklearn_info = st.sidebar.empty()

# sort_options = sorted(list(sort_func_dict.keys()) + [key + "_r" for key in sort_func_dict.keys() if key!="random"])
# sort_func = st.sidebar.selectbox("palette sort function", options=sort_options, index=5)

# # Random Number Seed
# seed = int(st.sidebar.number_input("random seed", value=42, help="Seed used for all random samplings."))
# np.random.seed(seed)
# st.sidebar.markdown("---")


# =======
#   App
# =======

file_type = "CSV"

def enqueue_output(file, queue):
    for line in iter(file.readline):
        queue.put(line)
    file.close()


def read_popen_pipes(p):

    with ThreadPoolExecutor(2) as pool:
        q_stdout, q_stderr = Queue(), Queue()

        pool.submit(enqueue_output, p.stdout, q_stdout)
        pool.submit(enqueue_output, p.stderr, q_stderr)

        while True:

            if p.poll() is not None and q_stdout.empty() and q_stderr.empty():
                break

            out_line = err_line = ''

            try:
                out_line = q_stdout.get_nowait()
            except Empty:
                pass
            try:
                err_line = q_stderr.get_nowait()
            except Empty:
                pass

            yield (out_line, err_line)

mean_standard = {'MI_dir_L5_weight': 93.26944471807772, 'MI_dir_L5_mean': 191.4107472827913, 'MI_dir_L5_variance': 18082.36561902601, 'MI_dir_L3_weight': 147.79771305906476, 'MI_dir_L3_mean': 191.78709449963588, 'MI_dir_L3_variance': 20115.809813276137, 'MI_dir_L1_weight': 422.69230621404853, 'MI_dir_L1_mean': 192.45398849572408, 'MI_dir_L1_variance': 21632.825106993005, 'MI_dir_L0.1_weight': 3956.491492914809, 'MI_dir_L0.1_mean': 193.0699092738356, 'MI_dir_L0.1_variance': 22237.066016856134, 'MI_dir_L0.01_weight': 23793.412747541897, 'MI_dir_L0.01_mean': 193.01654565510844, 'MI_dir_L0.01_variance': 22539.9332115765, 'H_L5_weight': 93.26944581738746, 'H_L5_mean': 191.41075164554763, 'H_L5_variance': 18082.366468211738, 'H_L3_weight': 147.79771466815876, 'H_L3_mean': 191.78710590340305, 'H_L3_variance': 20115.81218825291, 'H_L1_weight': 422.6923093676522, 'H_L1_mean': 192.45402679467415, 'H_L1_variance': 21632.832355214105, 'H_L0.1_weight': 3956.491520871961, 'H_L0.1_mean': 193.06988798218413, 'H_L0.1_variance': 22237.128611485674, 'H_L0.01_weight': 23793.41297913415, 'H_L0.01_mean': 193.01641747208495, 'H_L0.01_variance': 22540.042839561123, 'HH_L5_weight': 48.57348305538757, 'HH_L5_mean': 192.97834430973938, 'HH_L5_std': 1.7385065758928482, 'HH_L5_magnitude': 202.13845242297634, 'HH_L5_radius': 1015.1592102813747, 'HH_L5_covariance': -42.26661944608412, 'HH_L5_pcc': -0.00017076370448230644, 'HH_L3_weight': 78.8458688800357, 'HH_L3_mean': 192.94886524266582, 'HH_L3_std': 1.8975861392031632, 'HH_L3_magnitude': 202.14147930963802, 'HH_L3_radius': 1076.251172271592, 'HH_L3_covariance': -54.94590016968729, 'HH_L3_pcc': 7.17244348425036e-06, 'HH_L1_weight': 228.81715364268152, 'HH_L1_mean': 192.88272783991468, 'HH_L1_std': 2.4522577940438013, 'HH_L1_magnitude': 202.0854965121485, 'HH_L1_radius': 1157.949955032346, 'HH_L1_covariance': -71.42757321442832, 'HH_L1_pcc': 0.0007375210638640153, 'HH_L0.1_weight': 2080.7310406959023, 'HH_L0.1_mean': 192.91933177259202, 'HH_L0.1_std': 4.307469206483778, 'HH_L0.1_magnitude': 201.85908207058426, 'HH_L0.1_radius': 1455.4437111175168, 'HH_L0.1_covariance': -55.78491325953138, 'HH_L0.1_pcc': 0.006694613707242562, 'HH_L0.01_weight': 11026.991675978004, 'HH_L0.01_mean': 193.07143610205424, 'HH_L0.01_std': 5.520802243195934, 'HH_L0.01_magnitude': 201.941289227371, 'HH_L0.01_radius': 1800.2646738286505, 'HH_L0.01_covariance': 24.097176851819402, 'HH_L0.01_pcc': 0.012069709509734518, 'HH_jit_L5_weight': 48.57348305538757, 'HH_jit_L5_mean': 628687983.5799067, 'HH_jit_L5_variance': 112340819440481.78, 'HH_jit_L3_weight': 78.8458688800357, 'HH_jit_L3_mean': 628751145.8463289, 'HH_jit_L3_variance': 197641244861511.25, 'HH_jit_L1_weight': 228.8171536426819, 'HH_jit_L1_mean': 631157099.7313951, 'HH_jit_L1_variance': 3405251850774851.5, 'HH_jit_L0.1_weight': 2080.7310406959123, 'HH_jit_L0.1_mean': 643901119.4025023, 'HH_jit_L0.1_variance': 1.4570148806055658e+16, 'HH_jit_L0.01_weight': 11026.99167597793, 'HH_jit_L0.01_mean': 647138980.8789642, 'HH_jit_L0.01_variance': 1.6353972281457586e+16, 'HpHp_L5_weight': 4.155876790085, 'HpHp_L5_mean': 193.04962834021617, 'HpHp_L5_std': 1.0465706575651, 'HpHp_L5_magnitude': 200.1744897264621, 'HpHp_L5_radius': 599.9943653451037, 'HpHp_L5_covariance': 52.23516013649219, 'HpHp_L5_pcc': 0.0009395902603454411, 'HpHp_L3_weight': 6.114202414667558, 'HpHp_L3_mean': 193.03115333530116, 'HpHp_L3_std': 1.0882708603848092, 'HpHp_L3_magnitude': 200.17632192361475, 'HpHp_L3_radius': 652.1704052924346, 'HpHp_L3_covariance': 54.545628664337436, 'HpHp_L3_pcc': 0.001024547713182817, 'HpHp_L1_weight': 15.799919287955433, 'HpHp_L1_mean': 192.97496642227657, 'HpHp_L1_std': 1.1836082766974791, 'HpHp_L1_magnitude': 200.13403557831117, 'HpHp_L1_radius': 711.8888868541, 'HpHp_L1_covariance': 57.98777479891492, 'HpHp_L1_pcc': 0.0008930839307075592, 'HpHp_L0.1_weight': 142.17035985035372, 'HpHp_L0.1_mean': 192.87213318872682, 'HpHp_L0.1_std': 1.6271694757690296, 'HpHp_L0.1_magnitude': 199.81857577779715, 'HpHp_L0.1_radius': 942.5901007011578, 'HpHp_L0.1_covariance': 63.37947466920262, 'HpHp_L0.1_pcc': 0.0015511964391728128, 'HpHp_L0.01_weight': 809.4731319852671, 'HpHp_L0.01_mean': 192.76908789601808, 'HpHp_L0.01_std': 1.820125342690955, 'HpHp_L0.01_magnitude': 199.79717965555466, 'HpHp_L0.01_radius': 1261.4587873489922, 'HpHp_L0.01_covariance': 77.5912223343027}
std_standard = {'MI_dir_L5_weight': 65.40761114627169, 'MI_dir_L5_mean': 167.18636142867794, 'MI_dir_L5_variance': 25923.318386269715, 'MI_dir_L3_weight': 103.93752816203764, 'MI_dir_L3_mean': 161.14941602593373, 'MI_dir_L3_variance': 27186.26475124098, 'MI_dir_L1_weight': 300.74232681428316, 'MI_dir_L1_mean': 156.6659727830628, 'MI_dir_L1_variance': 28037.96437162659, 'MI_dir_L0.1_weight': 2830.405540052882, 'MI_dir_L0.1_mean': 155.26547351863826, 'MI_dir_L0.1_variance': 28313.83798639461, 'MI_dir_L0.01_weight': 20218.425533662754, 'MI_dir_L0.01_mean': 155.04640981959292, 'MI_dir_L0.01_variance': 27978.396451801727, 'H_L5_weight': 65.40760961166421, 'H_L5_mean': 167.18636536256307, 'H_L5_variance': 25923.317819656284, 'H_L3_weight': 103.9375259101526, 'H_L3_mean': 161.1494271222162, 'H_L3_variance': 27186.263166109795, 'H_L1_weight': 300.74232241340354, 'H_L1_mean': 156.66601331155354, 'H_L1_variance': 28037.959803860016, 'H_L0.1_weight': 2830.405500995867, 'H_L0.1_mean': 155.26513358818116, 'H_L0.1_variance': 28313.801818043023, 'H_L0.01_weight': 20218.42526116254, 'H_L0.01_mean': 155.04558179145764, 'H_L0.01_variance': 27978.330330981662, 'HH_L5_weight': 59.85275070845524, 'HH_L5_mean': 214.460293588465, 'HH_L5_std': 19.761329690953502, 'HH_L5_magnitude': 215.32578140500848, 'HH_L5_radius': 16874.37846281378, 'HH_L5_covariance': 2341.251085216031, 'HH_L5_pcc': 0.03167116955031993, 'HH_L3_weight': 97.38626974617111, 'HH_L3_mean': 214.3546851629615, 'HH_L3_std': 19.94646451124956, 'HH_L3_magnitude': 215.2227234995502, 'HH_L3_radius': 17612.773354575864, 'HH_L3_covariance': 2591.317219594441, 'HH_L3_pcc': 0.03675175653358045, 'HH_L1_weight': 283.61450698767186, 'HH_L1_mean': 214.21400221355728, 'HH_L1_std': 20.20883373338824, 'HH_L1_magnitude': 215.06417214457423, 'HH_L1_radius': 18252.492709663617, 'HH_L1_covariance': 2851.96284197137, 'HH_L1_pcc': 0.049585684789514924, 'HH_L0.1_weight': 2531.2952547805357, 'HH_L0.1_mean': 214.09245572224572, 'HH_L0.1_std': 22.700448615886838, 'HH_L0.1_magnitude': 214.38883333782934, 'HH_L0.1_radius': 19198.99644085447, 'HH_L0.1_covariance': 2773.986437822168, 'HH_L0.1_pcc': 0.08055039304760538, 'HH_L0.01_weight': 14382.975460265521, 'HH_L0.01_mean': 214.1395784684696, 'HH_L0.01_std': 25.348287398534648, 'HH_L0.01_magnitude': 214.0047239780351, 'HH_L0.01_radius': 19339.88973720824, 'HH_L0.01_covariance': 2253.7581059384424, 'HH_L0.01_pcc': 0.09580042702307627, 'HH_jit_L5_weight': 59.85275070845524, 'HH_jit_L5_mean': 743016473.5743761, 'HH_jit_L5_variance': 6250734702901485.0, 'HH_jit_L3_weight': 97.38626974617115, 'HH_jit_L3_mean': 742969733.7671242, 'HH_jit_L3_variance': 6950980850854370.0, 'HH_jit_L1_weight': 283.61450698767254, 'HH_jit_L1_mean': 741207524.8975542, 'HH_jit_L1_variance': 2.7593568000854604e+16, 'HH_jit_L0.1_weight': 2531.295254780552, 'HH_jit_L0.1_mean': 735639959.680993, 'HH_jit_L0.1_variance': 8.302410347809195e+16, 'HH_jit_L0.01_weight': 14382.975460265423, 'HH_jit_L0.01_mean': 734900370.1353989, 'HH_jit_L0.01_variance': 9.062917156568589e+16, 'HpHp_L5_weight': 14.916354189488633, 'HpHp_L5_mean': 214.90560246281223, 'HpHp_L5_std': 16.31536442623088, 'HpHp_L5_magnitude': 216.5048671747931, 'HpHp_L5_radius': 11731.921287501496, 'HpHp_L5_covariance': 1355.550615852965, 'HpHp_L5_pcc': 0.031559148213155364, 'HpHp_L3_weight': 23.813137037069808, 'HpHp_L3_mean': 214.77739782553834, 'HpHp_L3_std': 16.977476241343275, 'HpHp_L3_magnitude': 216.40067325338478, 'HpHp_L3_radius': 12738.551333658606, 'HpHp_L3_covariance': 1441.8509375716196, 'HpHp_L3_pcc': 0.03536669461717413, 'HpHp_L1_weight': 67.87674065863807, 'HpHp_L1_mean': 214.5814633265508, 'HpHp_L1_std': 17.771723529853336, 'HpHp_L1_magnitude': 216.23883366182812, 'HpHp_L1_radius': 13903.015496614113, 'HpHp_L1_covariance': 1596.845577880396, 'HpHp_L1_pcc': 0.04371894878750802, 'HpHp_L0.1_weight': 641.8053849642735, 'HpHp_L0.1_mean': 214.34376453013917, 'HpHp_L0.1_std': 18.8604528320339, 'HpHp_L0.1_magnitude': 215.67686798817374, 'HpHp_L0.1_radius': 15893.714854713837, 'HpHp_L0.1_covariance': 1878.607200061036, 'HpHp_L0.1_pcc': 0.05649817929768701, 'HpHp_L0.01_weight': 3880.2362089838475, 'HpHp_L0.01_mean': 214.1068456947141, 'HpHp_L0.01_std': 18.309516854040915, 'HpHp_L0.01_magnitude': 215.2659136261106, 'HpHp_L0.01_radius': 17747.069353764848, 'HpHp_L0.01_covariance': 2453.3182055179027}

#standardize numerical columns
def standardize(df,col):
    df[col]= (df[col]-mean_standard[col])/std_standard[col]


model = Net(115, 10)
model.load_state_dict(torch.load("IoT_Intrusions_Detection.pth"))
model.eval()

file_data = None
df = None

def run_and_display_stdout():
    # while True:
    #     result = subprocess.Popen(cmd_with_args, stdout=subprocess.PIPE)
    #     for line in iter(lambda: result.stdout.readline(), b""):
    #         st.write(line.decode("utf-8"))
    #     time.sleep(1)

    with sp.Popen("python test.py", stdout=sp.PIPE, stderr=sp.PIPE, text=True) as p:

        for out_line, err_line in read_popen_pipes(p):

            # Do stuff with each line, e.g.:
            st.write(out_line)
            st.write(err_line)

        return p.poll() # return status-code

def run_and_display_stdoutc(*cmd_with_args):
    while (True):
        i += 1
        status_woker = "TT"
        # for worker in workers:
        #     if (workers[worker]["is_active"]):
        #         # Replace the placeholder with some text:
        #         status_woker += "🟢" + "(" + str(worker[0]) + "," + str(worker[1]) + ")" + "\n"
        #     else:
        #         status_woker += "🟤" + "(" + str(worker[0]) + "," + str(worker[1]) + ")" + "\n"
        print("\n\n", status_woker, "\n\n")
        st.write(status_woker)
        # placeholder.text(status_woker)
        placeholder.text(i)
        time.sleep(5)
    # while True:
    #     print(i)
    #     i += 1
    #     st.markdown("##### 📌 Upload CSV file ")
    #     st.markdown(i)
    #     time.sleep(2)
    # st.write("TT")
    # result = subprocess.Popen(cmd_with_args, stdout=subprocess.PIPE)
    # for line in iter(lambda: result.stdout.readline(), b""):
    #     st.write(line.decode("utf-8"))

# provide options to either select an image form the gallery, upload one, or fetch from URL

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

aboutus_tab,  upload_tab, realtime_monitor = st.tabs(["ABOUT US", "PREDICT", "REALTIME MONITOR"])
with aboutus_tab:

    url_text = st.empty()
    
    # FIXME: the button is a bit buggy, but it's worth fixing this later

    # url_reset = st.button("Clear URL", key="url_reset")
    # if url_reset and "image_url" in st.session_state:
    #     st.session_state["image_url"] = ""
    #     st.write(st.session_state["image_url"])

    url = url_text.text_input("Image URL", key="image_url")
    
    if url!="":
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
        except:
            st.error("The URL does not seem to be valid.")
   
    

with upload_tab:
    # file = st.file_uploader("Upload Art", key="file_uploader")
    # if file is not None:
    #     try:
    #         img = Image.open(file)
    #     except:
    #         st.error("The file you uploaded does not seem to be a valid image. Try uploading a png or jpg file.")
    # if st.session_state.get("image_url") not in ["", None]:
    #     st.warning("To use the file uploader, remove the image URL first.")

    st.markdown("")

    st.markdown("##### 📌 Upload CSV file ")

    threat_color  = {
        "benign": "white" ,
        "gafgyt_combo": "LightSteelBlue",
        "gafgyt_junk": "Plum",
        "gafgyt_scan": "LightPink",
        "gafgyt_udp": "Khaki",
        "mirai_ack": "BurlyWood",
        "mirai_scan": "Coral",
        "mirai_syn": "DarkGray",
        "mirai_udp": "SlateGray",
        "mirai_udpplain": "IndianRed"
    }

    if file_type == "CSV":
        file = st.file_uploader("Upload CSV file", type="csv")
        if file:
            df = pd.read_csv(file)
            st.write("Uploaded CSV file:")
            st.write(df)       
            file_data = df.to_csv(index=False)

            data_st = df
            print(data_st)
            for i in (data_st.iloc[:,:-1].columns):
                standardize(data_st,i)

            X_test = data_st.values


            n_feature = X_test.shape[1]


            print("Number of testing features : ", n_feature)


            BATCH_SIZE = 1000

            # Create pytorch tensor from X_test, y_test
            test_inputs = torch.tensor(X_test,dtype=torch.float)


            dataset = TensorDataset(test_inputs)
            data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

            encoder = LabelEncoder()
            encoder.fit_transform(['benign', 'gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_udp', 'mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udp', 'mirai_udpplain'])
            # Take the random record from the test data
            if st.button("Predict"):
                result = []
                
                for data in test_inputs:
                    data_reshape = data.reshape(1, n_feature, 1)
                    pred = model(data_reshape)
                    pred_label = int(pred.argmax().data.cpu().numpy())
                    pred_threat = encoder.inverse_transform([pred_label])[0]
                    result.append(pred_threat)
                    
                # result = ['gafgyt_combo', 'mirai_udp', 'mirai_udp', 'mirai_udp', 'mirai_udp', 'mirai_udp', 'mirai_udp', 'mirai_udp']
                df["pred"] = result
                st.write("Predicted threat type: ")   
                for idx in range(0,df.shape[0],1):
                    # df.at[idx, 'affiliate_id'] = df['affiliate_id'].iloc[idx] + 1
                    q = df.at[idx,'pred']
                    t = f"<span class='highlight {threat_color[df.at[idx,'pred']]}'> [ Packet {idx} ] {q}  <span class='bold'></span>"
                    st.markdown(t, unsafe_allow_html=True)  


            st.markdown("")
            st.markdown("##### 🎈 Check & download results ")

            st.header("")

            cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

            with c1:
                CSVButton2 = download_button(df, "Data.csv", "📥 Download (.csv)")
            with c2:
                CSVButton2 = download_button(df, "Data.txt", "📥 Download (.txt)")
            
            st.header("")

    st.markdown("")
    st.markdown("---")
    st.markdown("")
    st.markdown("<p style='text-align: center'><a href='https://github.com/Kaludii'>Github</a> | <a href='https://huggingface.co/Kaludi'>HuggingFace</a></p>", unsafe_allow_html=True)

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with realtime_monitor:
    if st.button("Monitor"):
        main()
    
# convert RGBA to RGB if necessary
# n_dims = np.array(img).shape[-1]
# if n_dims == 4:
#     background = Image.new("RGB", img.size, (255, 255, 255))
#     background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
#     img = background

# apply image enhancements
# for cat in enhancement_categories:
#     img = getattr(ImageEnhance, cat)(img)
#     img = img.enhance(enhancement_factor_dict[cat])

# # show the image
# with st.expander("🖼  Artwork", expanded=True):
#     st.image(img, use_column_width=True)


# if click or toggle:
    
#     # df_rgb = get_df_rgb(img, sample_size)

#     # (optional for later)
#     # plot_rgb_3d(df_rgb) 
#     # plot_hsv_3d(df_rgb) 

#     # calculate the RGB palette and cache it to session_state
#     # st.session_state["palette_rgb"] = get_palette(df_rgb, model_name, palette_size, sort_func=sort_func)

#     if "palette_rgb" in st.session_state:
        
#         # store individual colors in session state
#         store_palette(st.session_state["palette_rgb"])

#         st.write("---")

#         # sort the colors based on the selected option
#         colors = {k: v for k, v in st.session_state.items() if k.startswith("col_")}
#         sorted_colors = {k: colors[k] for k in sorted(colors, key=lambda k: int(k.split("_")[-1]))}
        
#         # find the hex representation for matplotlib and plotly settings
#         palette_hex = [color for color in sorted_colors.values()][:palette_size]
#         with st.expander("Adopt this Palette", expanded=False):
#             st.pyplot(show_palette(palette_hex))

#             matplotlib_tab, plotly_tab = st.tabs(["matplotlib", "plotly"])

#             with matplotlib_tab:
#                 display_matplotlib_code(palette_hex)

#                 import matplotlib as mpl
#                 from cycler import cycler

#                 mpl.rcParams["axes.prop_cycle"] = cycler(color=palette_hex)
#                 import matplotlib.pyplot as plt

#                 x = np.arange(5)
#                 y_list = np.random.random((len(palette_hex), 5))+2
#                 df = pd.DataFrame(y_list).T

#                 area_tab, bar_tab = st.tabs(["area chart", "bar chart"])

#                 with area_tab:
#                     fig_area , ax_area = plt.subplots()
#                     df.plot(kind="area", ax=ax_area, backend="matplotlib", )  
#                     st.header("Example Area Chart")
#                     st.pyplot(fig_area)
    
#                 with bar_tab:
#                     fig_bar , ax_bar = plt.subplots()
#                     df.plot(kind="bar", ax=ax_bar, stacked=True, backend="matplotlib", )
#                     st.header("Example Bar Chart")
#                     st.pyplot(fig_bar)

                
#             with plotly_tab:
#                 display_plotly_code(palette_hex)

#                 import plotly.io as pio
#                 import plotly.graph_objects as go
#                 pio.templates["sophisticated"] = go.layout.Template(
#                     layout=go.Layout(
#                     colorway=palette_hex
#                     )
#                 )
#                 pio.templates.default = 'sophisticated'

#                 area_tab, bar_tab = st.tabs(["area chart", "bar chart"])

#                 with area_tab:
#                     fig_area = df.plot(kind="area", backend="plotly", )
#                     st.header("Example Area Chart")
#                     st.plotly_chart(fig_area, use_container_width=True)
    
#                 with bar_tab:
#                     fig_bar = df.plot(kind="bar", backend="plotly", barmode="stack")
#                     st.header("Example Bar Chart")
#                     st.plotly_chart(fig_bar, use_container_width=True)

       
# else:
#     st.info("👈  Click on 'Find Palette' ot turn on 'Toggle Update' to see the color palette.")

# st.sidebar.success(print_praise())   

# if __name__ == "__main__":
#     st.write("QQ")
#     main()
