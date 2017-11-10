'''
This script procces the json files that have been produced by an hierachical clutering algorithm
using CNN descriptors in order for smoothing the result of the clustering algorithm.

inputs:
        --datasePath Path of the directory SmartCity_<CITYNAME> where the results will be store.
        --cnnModelName CNN Model used for obtaining the the CNN descriptors of the images.
output:
        --json file for every map that include the calculated values for each cluster in the generated map.
           the file will be placed in datasetPath/CNNMODEL/mapProcessed/mapname.json



Elimina del la lista de coordenadas el valor del punto que estoy analizando e instancia un nuevo clasificador, para buscar los mas cercanos
Emplea una ponderacion de distacia para los pesos que le asigne al los puntos.

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
from sklearn.neighbors import KNeighborsClassifier



class CoordinatePoint:
    lat=0
    lon=0


#arguments parser object
parser = argparse.ArgumentParser(description='Clustering of CNN Descriptors.', prog='clustering_script.py')
parser.add_argument("-datapath", help="path to the directory of the dataset in the computer", action="store", metavar="/home/user/documents/SmartCity_Name")
parser.add_argument("-cnnmodel", help="CNN model Name used for the experiments", action="store", metavar="ImageNet")

args = parser.parse_args()

if not len(sys.argv) > 1:
	print("Please Introduce the required values.")
	parser.print_help()
	exit(0)


#argument Parse
datasetPath=  "/home/jcarlos2289/Documentos/SmartCity_SanVicente" #args.datapath# "/home/jcarlos2289/Documentos/SmartCity_SanVicente"
cnnModelName= "Places205"#args.cnnmodel#"Places205"

mapPaths="/home/jcarlos2289/Documentos/python_ws/knn/"  #datasetPath +"/"+cnnModelName+"/"+"generatedMapsPaths/"

#obtaining the list of generated maps in the dataset directory
mapList=os.listdir(mapPaths)
mapGeneratedPath = datasetPath +"/"+cnnModelName+"/generatedMaps/"
mapGeneratedPathPth = datasetPath +"/"+cnnModelName+"/generatedMapsPaths" 



#mapPathFiles=open('ListaMapsPaths.txt', 'r') #['generatedMapsPaths/clusterListPaths_average_braycurtis_0.6_Reduced.json']#
k=3
clusterDic={}
coordList = []
categoryList=[]
for map in mapList:
        if map.endswith(".json"):
                fullClusterList=[]
                clusterDic={}
                mapPth=mapPaths+map.replace("\n", "")
                
                with open(mapPth) as data_file:    
                        generatedMap=  json.load(data_file)
                

                for key, value in generatedMap.iteritems():
                        coordFullList = value
                        for val in value:
                                #print(val['latitude'])
                                coordArr=([val["latitude"],  val["longitude"]])  #np.array
                                coordList.append(coordArr)
                                categoryList.append(key)

                                #print(len(coordList))
                                #print(len(categoryList))
            
                

                predictionFilePath = "PredictionFile.txt"
                logFile = open(predictionFilePath, 'w+')
                logFile.write("Predicted\tGT\n")
                logFile.close()

                logFile = open(predictionFilePath, 'a+')
                cluster_dict = defaultdict(list)
                cluster_dictPath =defaultdict(list)

                
                #neigh.fit(coordList,categoryList)
                i = 0
                for x, y in zip(coordList, categoryList):
                        coordDict={}
                        coordDict["latitude"]=x[0]
                        coordDict["longitude"]=x[1]
                        #print(x)
                        valIndex = coordList.index(x)
                        #print("Val index")
                        #print(valIndex)

                        xFix=coordList[:]
                        yFix=categoryList[:]

                        del xFix[valIndex]
                        del yFix[valIndex]

                        neigh = KNeighborsClassifier(n_neighbors=k, weights='distance')
                        neigh.fit(xFix,yFix)
                        if i%100==0:
                            print(i)
                        i= i+1
                        #x = x.reshape(1,-1)
                        fx = neigh.predict(np.asarray(x).reshape(1,-1))
                        #print(fx)
        
                        

                        #filename
                        fileName=datasetPath+"/"+cnnModelName+"/tagsReduced/lat="+coordDict["latitude"].replace(".",",") +"_long="+ coordDict["longitude"].replace(".",",") +".txt"

                        #x = x.reshape(1,-1)
                        #print(x)
                        #fx = neigh.predict(x)
                        cluster_dict[fx[0]].append(coordDict)
                        cluster_dictPath[fx[0]].append(fileName)

                        #print(fx)
                        logFile.write( str(fx[0]) +"\t"+ str(y)+" \n")
                logFile.close()

                flName = map.replace(".json","_knn_"+str(k)+".json" )

                with open(mapGeneratedPath+"/"+flName, 'w') as fp:  # clusterList_average_braycurtis_0.6_"+"knn"+str(k)+".json
                            json.dump(cluster_dict, fp, sort_keys=True )

                pthFlName = flName.replace("List","ListPaths")
                with open(mapGeneratedPathPth+"/"+pthFlName, 'w') as fp:
                            json.dump(cluster_dictPath, fp, sort_keys=True, indent=4 ) #clusterListPaths_average_braycurtis_0.6_"+"knn"+str(k)+".json

#print( neigh.predict([[38.39523, -0.52098]]))
#print(neigh.predict_proba([[38.39523, -0.52098]]))


#arreglar el nombre del fichero a crear para auq solo se agrege el knn#
