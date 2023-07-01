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


#standardize numerical columns
def standardize(df,col):
    df[col]= (df[col]-df[col].mean())/df[col].std()

model_new = Net(115, 10)
model_new.load_state_dict(torch.load("IoT_Intrusions_Detection.pth"))
model_new.eval()

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

aboutus_tab,  upload_tab, realtime_monitor = st.tabs(["ABOUT US", "PREDICT", "Realtime Monitor"])
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
                result = subprocess.run(["sudo", "python3", "sniffer.py", "-i", "ens33", "output", "--flow"])
                st.write(result)
                print("SAAS: ", result)
                # for data in test_inputs:
                #     data_reshape = data.reshape(1, n_feature, 1)
                #     pred = model_new(data_reshape)
                #     pred_label = int(pred.argmax().data.cpu().numpy())
                #     pred_threat = encoder.inverse_transform([pred_label])[0]
                #     result.append("mirai_udpplain")
                    
                # result = []
                # result = ['gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_udp', 'mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udp', 'mirai_udpplain']
                # df["pred"] = result
                # st.write("Predicted threat type: ")   
                # for idx in range(0,df.shape[0],1):
                #     # df.at[idx, 'affiliate_id'] = df['affiliate_id'].iloc[idx] + 1
                #     q = df.at[idx,'pred']
                #     t = f"<span class='highlight {threat_color[df.at[idx,'pred']]}'> [ Packet {idx} ] {q}  <span class='bold'></span>"
                #     st.markdown(t, unsafe_allow_html=True)  


            st.markdown("")
            st.markdown("##### 🎈 Check & download results ")

            st.header("")

            cs, c1, c2, c3, cLast = st.columns([2, 2, 2, 1.5, 2])

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


















































