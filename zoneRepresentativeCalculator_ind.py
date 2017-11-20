'''
This script procces the json files that have been produced by an hierachical clutering algorithm
using CNN descriptors in order for processing the images that belong to the differrnts clusters.
After that calculate for every cluster their mean, varince and standard deviation and store them
in a json file with the information of every cluster.

inputs:
        --datasePath Path of the directory SmartCity_<CITYNAME> where the results will be store.
        --cnnModelName CNN Model used for obtaining the the CNN descriptors of the images.
output:
        --json file for every map that include the calculated values for each cluster in the generated map.
           the file will be placed in datasetPath/CNNMODEL/mapProcessed/mapname.json
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, is_valid_linkage
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import json
from collections import defaultdict
from pprint import pprint
import os
import argparse
import sys 

#arguments parser object
parser = argparse.ArgumentParser(description='Clustering of CNN Descriptors.', prog='clustering_script.py')
parser.add_argument("-datapath", help="path to the directory of the dataset in the computer", action="store", metavar="/home/user/documents/SmartCity_Name")
parser.add_argument("-cnnmodel", help="CNN model Name used for the experiments", action="store", metavar="ImageNet")
parser.add_argument("-measure", help="Measure for Experiment 1 for mean, 2 for median", action="store", metavar="1")

args = parser.parse_args()

if not len(sys.argv) > 1:
	print("Please Introduce the required values.")
	parser.print_help()
	exit(0)


#argument Parse
datasetPath="/home/jcarlos2289/Documentos/SmartCity_SanVicente" #args.datapath
cnnModelName= "Places205" #args.cnnmodel

if int(args.measure) ==1:
        measure=""
elif int(args.measure) ==2:
        measure="Median"
else:
        parser.print_help()
        print("Provide Right Measure Parameter")
	exit(0)
        

mapPaths=datasetPath +"/"+cnnModelName+"/"+"generatedMapsPaths"+measure+"/"

#obtaining the list of generated maps in the dataset directory
mapList=os.listdir(mapPaths)
mapProcessedPath = datasetPath +"/"+cnnModelName+"/mapsProcessed"+measure+"/"

mapList.append("clusterListPaths_average_braycurtis_0.6_Cknn_25.json")

#mapPathFiles=open('ListaMapsPaths.txt', 'r') #['generatedMapsPaths/clusterListPaths_average_braycurtis_0.6_Reduced.json']#
clusterDic={}
for map in mapList:
        if map.endswith(".json"):
                fullClusterList=[]
                clusterDic={}
                mapPth=mapPaths+map.replace("\n", "")
                
                with open(mapPth) as data_file:    
                        generatedMap=  json.load(data_file)
                

                for key, value in generatedMap.iteritems():
                        PathListFile = value
                        #print('Cluster {0}-Len {1}'.format(key, len(value)))
                        ImageList=[]
                        for filePath in PathListFile: 
                                file = open(filePath.replace("\n", ""), 'r') 
                                dict ={}
                                for line in file: 
                                        data= str.split(line,'-')
                                        dict[data[0]]= float(data[1])
                                ImageList.append(dict)
                        #Convierte la Lista de Diccionarios en un Numpy Array, ordena las Keys en orden alfabetico
                        ImageDescriptorArray = pd.DataFrame(ImageList)
                        #ImageMatrix =ImageDescriptorArray.values  #comented 28/10/2017
                      
                        var=ImageDescriptorArray.var(ddof=0)
                        mean=ImageDescriptorArray.mean()
                        stdDev =ImageDescriptorArray.std(ddof=0) 
                       
                        clusterDataDic={}
                        clusterDataDic["mean"]=mean.round(6).to_dict()
                        clusterDataDic["var"]=var.round(6).to_dict()
                        clusterDataDic["desv"]=stdDev.round(6).to_dict()
                        clusterDataDic["lenght"]=len(value)
                        
                        clusterDic[key]=clusterDataDic
                        ##fullClusterList.append(clusterDic)
                
                #fileName_0=mapPth.split("/")
                #fileName_1=fileName_0[len(fileName_0)-1]
                #fileName=fileName_1.replace("ListPaths","Stats")
                fileName=map.replace("ListPaths","Stats")
                #print(fileName)
                savingPath=mapProcessedPath
                if not os.path.exists(savingPath):
                        os.makedirs(savingPath)
                
                saveFilePath=mapProcessedPath+fileName
                #print(saveFilePath)
                with open(saveFilePath, 'w') as fp:
                        json.dump(clusterDic, fp, sort_keys=True) #, indent=4
                with open("mapsProcessed/"+fileName, 'w') as fp:
                        json.dump(clusterDic, fp, sort_keys=True) #, indent=4
                


