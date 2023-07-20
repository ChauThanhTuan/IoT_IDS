
# Federated IDS For IoT  
<i> This project is our graduation thesis approved in July 2023 by the board of University of Information Technology </i>

<a href="https://nc.uit.edu.vn/">
<p align="center">
  <img width="600" height="250" src="https://nc.uit.edu.vn/wp-content/uploads/2019/08/logoncuit-2.png">
</p>
</a>

## Contents

- [Overview](#-overview)
- [Requirements](#-requirements)
- [Implementation](#-implementation)
- [Contribute](#-contribute)


## 📖 Overview
- We propose an IDS for IoT system that applies Feaderated Learning, which can be experimentally deployed on 3 IoT devices (including 1 for model synthesis and 2 workers for model training). During training, the worker saves the parameters of the best model in the corresponding `IoT_Intrusions_Detection<ID>.pth` file.
- In addition, to reduce the number of transfer times between Master and Worker, we provide two model optimization mechanisms which are Loss-based and Acc-based model transfers during training.
- The system can record training log and send it directly to the ELK system for real-time monitoring.
- Finally, a web interface is provided using the trained model to visualize the experimental phase. This website will load model parameters from IoT_Intrusions_Detection.pth file to make predictions based on user input, including 2 main features:
    - Prediction based on available network traffic file (.csv)
    - Real-time monitoring to predict network traffic captured from the device's network card.


## 📋 Requirements
  - PyTorch (v1.18)
  - Numpy (v1.19.5)
  - Scipy (v1.6.0)
  - Pandas (v1.3.5)
  - Elasticsearch (v7.17.9)
  - Scikit-learn
  - Matplotlib
  - Streamlit (v1.22.0)


## 🎉Implementation

<i>In this project, we use 3 `Raspberry Pi4, Ubuntu Server 22.10 (64-bit), docker (ubuntu 22.10)` for implementation.</i>

To start the training process, begin with:
```
$ git clone https://github.com/ChauThanhTuan/IoT_IDS.git
```
Configure IP, port in Master and Worker files corresponding to usage needs and continue to execute the following commands:
```
$ python Master.py 
$ python Worker1.py
$ python Worker2.py
```
or:
```
$ python Master.py 
$ python Worker1_OptimizeByAcc.py
$ python Worker2_OptimizeByAcc.py
```
or:
```
$ python Master.py 
$ python Worker1_OptimizeByLoss.py
$ python Worker2_OptimizeByLoss.py
```
To use trained model through web interface, run:
```
$ cd IoT_IDS
$ streamlit run sniffer.py
```
Go to the Local URL: http://localhost:8501 or the Network URL displayed on the console when running the application.

## 👏 Contribute

  <a href="https://github.com/ChauThanhTuan">Châu Thanh Tuấn</a>

  <a href="https://github.com/NguyenMyQuynh">Nguyễn Mỹ Quỳnh</a>


