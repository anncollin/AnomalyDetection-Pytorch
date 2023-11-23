# Anomaly Detection Framework for Industrial Vision (PyTorch)

## Context 

This repository contains the code related to our anomaly detection framework that uses an autoencoder trained on images corrupted with our Stain-shaped noise. The full paper is available on [ArXiv](https://arxiv.org/abs/2008.12977) and has been presented at ICPR2020. 
This repository contains the PyTorch-version of the intial tensorflow-version of the code available [here](https://github.com/anncollin/AnomalyDetection-Keras).

## Dependencies
Our implementation is built on PyTroch (version 1.11.0) and python 3.9. 

## Quick start
<ol>
  <li>Clone the repository and activate a compatible python environmnent. </li>
  <li>Download the MVTec AD dataset from their <a href="https://www.mvtec.com/company/research/datasets/mvtec-ad"> Website. </a> </li> Unzip it in the location of your choice.
  <li>Copy the absolute path to the MVTec AD dataset folder in the 'code/configs/server/datasetLocation.yaml' file.</li>
</ol> 

### First argument : train. 
Weights of our networks are provided in the repository. You can choose to use pretrained network to reproduce results by lanching: <br>
`python main.py`<br>
or to retrain networks by lanching: <br>
`python main.py -train true` <br>

### Second argument : exp. 
You can change multiple parameters by creating new configuration files in the 'code/Todo_list' folder. <br>

**Todo_List/default.yaml** <br>
By default, it will run the AESc network trained with our Stain noise model over the entire dataset: <br>
`python main.py`<br>

**Todo_List/AE_STAIN.yaml** <br>
It will run the AE network trained with our Stain noise model over the entire dataset: <br>
`python main.py -exp AE_STAIN `<br>

**Todo_List/AESc_noDataAugmentation.yaml** <br>
It will run the AESc network trained without data augmentation over the entire dataset: <br>
`python main.py -exp AE_STAIN `<br>

**Todo_List/AE_noDataAugmentation.yaml** <br>
It will run the AE network trained without data augmentation over the entire dataset: <br>
`python main.py -exp AE_STAIN `<br>
