# AdaRNN: Adaptive Learning and Forecasting for Time Series
**copy from https://github.com/jindongwang/transferlearning/tree/master/code/deep/adarnn**

## Requirement

- CUDA 10.1 
- Python 3.7.7
- Pytorch == 1.6.0
- Torchvision == 0.7.0

The required packages are listed in `requirements.txt`. 


## Dataset 

The original air-quality dataset is downloaded from [Beijing Multi-Site Air-Quality Data Data Set](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data) . The air-quality dataset contains hourly air quality information collected from 12 stations in Beijing from 03/2013 to 02/2017. We randomly chose four stations (Dongsi, Tiantan, Nongzhanguan, and Dingling) and select six features (PM2.5, PM10, S02, NO2, CO, and O3). Since there are some missing data, we simply fill the empty slots using averaged values. Then, the dataset is normalized before feeding into the network to scale all features into the same range. This process is accomplished by max-min normalization and ranges data between 0 and 1. The processed  air-quality dataset can be downloaded at [dataset link](https://box.nju.edu.cn/f/2239259e06dd4f4cbf64/?dl=1). 

The procossed .pkl files contains three arraies: 'feature', 'label', and 'label_reg'. 'label' refers to the classification label of air quality (e.g. excellence, good, middle), which is not used in this work and could be ignored. 'label_reg' refers to the prediction value.


## How to run

The code for air-quality dataset is in `train_weather.py`. After downloading the dataset, you can change args.data_path in `train_weather.py` to the folder where you place the data.

Then you can run the code. Taking Dongsi station as example, you can run 

`python3 train_weather.py --model_name 'AdaRNN' --station 'Dongsi' --pre_epoch 40 --dw 0.5 --loss_type 'adv' --data_mode 'tdc' --data_path dataset`

For transformer model, the adapted transformer model is in `transformer_adapt.py`, you can run,
`python3 transformer_adapt.py  --station 'Tiantan' --dw 1.0`

