import argparse
import yaml
import os
from os.path import dirname, abspath
import importlib
import itertools
from termcolor import colored
from multiprocessing import Process

from torchsummary import summary

from models.SuperClass import *
from datasets.dataAugmentation import *


rootPath = dirname(dirname(abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('-exp', type=str, default='DefaultExp')
parser.add_argument('-train', default=False, type=lambda x: (str(x).lower() == 'true'))

######################################################################################
#
# MAIN PROCEDURE 
# launches experiments whose parameters are described in a yaml file  
# 
# Example of use in the terminal: 
# python main.py -exp DefaultExp
# with 'DefaultExp' beeing the name of the yaml file (in the Todo_list folder) with 
# the wanted configuration 
# 
# By using: 
# python main.py -exp DefaultExp -train true
# You can (re-)train the network instead of importing pretrained weights (default)
######################################################################################

def main(parser):
    # -----------------
    # 0. INITIALISATION 
    # -----------------
    # Read the yaml configuration file 
    stream = open('Todo_List/' + parser.exp + '.yaml', 'r')
    args   = yaml.safe_load(stream)

    # Get all experiments parameter
    imgCategory   = args['DatasetFolder']['subfolder']
    datasetConfig = args['Dataset']
    modelConfig   = args['Model']
    trainConfig   = args['Train']
    dataAugConfig = args['DataAugmentation']
    serverConfig  = args['Server']

    allConfigs   = [imgCategory, datasetConfig, modelConfig, trainConfig, dataAugConfig]
    allExps      = list(itertools.product(*allConfigs))

    def call_training(thisExp):
        (thisImgCat, thisDS, thisModel, thisTrain, thisDA) = thisExp
        serverStream = open('configs/server/' + serverConfig[0] + '.yaml', 'r')
        serverCfg    = yaml.safe_load(serverStream)
        folderPath   = serverCfg['DatasetLocation'] + '/' + thisImgCat

        if not os.path.exists(folderPath):
            print('ERROR : Unknown dataset location. Please update the "code/configs/server/datasetLocation.yaml" file')
            exit(1)


        # ------------------------
        # 1. NETWORK INSTANTIATION 
        # ------------------------
        modelStream     = open('configs/model/' + thisModel + '.yaml', 'r')
        thisModelConfig = yaml.safe_load(modelStream)
        if thisModelConfig['Model_class'] == 'Super': 
            myNet = Network_Class(thisImgCat, thisDS, thisModel, thisTrain, thisDA, folderPath)
        else: 
            module = importlib.import_module('models.'+ thisModelConfig['Model_class'] +'Class')
            class_ = getattr(module, thisModelConfig['Model_class']) 
            myNet  = class_(thisImgCat, thisDS, thisModel, thisTrain, thisDA, folderPath)
        print(colored('CURRENT EXPERIMENT :  ' + thisImgCat + ' : ' + myNet.expName, 'red'))
        #summary(myNet.model, myNet.model.inputDim)

        # ------------------
        # 2. TRAIN THE MODEL  
        # ------------------
        resultPath = rootPath + '/Results/' + myNet.expName +'/' + myNet.imgCat
        if parser.train:
            print(colored('Start to train the network', 'red'))
            myNet.train(resultPath)
            print(colored('The network is trained', 'red'))

        # ---------------------
        # 3. EVALUATE THE MODEL  
        # ---------------------  
        myNet.loadWeights(resultPath + '/_Weights/wghts.pkl')
        myNet.evaluate(resultPath = resultPath, printPrediction = True)


    for thisExp in allExps:
        p = Process(target=call_training, args=(thisExp,))
        p.start()
        p.join()


if __name__ == '__main__':
    parser = parser.parse_args()
    main(parser)