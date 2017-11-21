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
    
Ejecuta el Knn tres veces sobre la lista de coordeandas
En cada iteracion se va alamcenando el cluster asisgando al punto en una una lista de puntos y clusters, en la siguiente ejecucion completa del KNN se dispondra
de una lista actualizada de las categorias de cada coordenada.

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
parser.add_argument("-measure", help="Measure for Experiment 1 for mean, 2 for median", action="store", metavar="1")


args = parser.parse_args()

if not len(sys.argv) > 1:
	print("Please Introduce the required values.")
	parser.print_help()
	exit(0)


#argument Parse
datasetPath=  "/home/jcarlos2289/Documentos/SmartCity_SanVicente" #args.datapath# "/home/jcarlos2289/Documentos/SmartCity_SanVicente"
cnnModelName= "Places205"#args.cnnmodel#"Places205"

if int(args.measure) ==1:
        measure=""
elif int(args.measure) ==2:
        measure="_Median"
else:
        parser.print_help()
        print("Provide Right Measure Parameter")
	exit(0)
        




mapPaths="/home/jcarlos2289/Documentos/python_ws/knn/"  #datasetPath +"/"+cnnModelName+"/"+"generatedMapsPaths/"

#obtaining the list of generated maps in the dataset directory
mapList=os.listdir(mapPaths)
mapGeneratedPath = datasetPath +"/"+cnnModelName+"/generatedMaps/"
mapGeneratedPathPth = datasetPath +"/"+cnnModelName+"/generatedMapsPaths"



#mapPathFiles=open('ListaMapsPaths.txt', 'r') #['generatedMapsPaths/clusterListPaths_average_braycurtis_0.6_Reduced.json']#
k=25
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
            
                

                '''predictionFilePath = "PredictionFile.txt"
                logFile = open(predictionFilePath, 'w+')
                logFile.write("Predicted\tGT\n")
                logFile.close()'''

                '''logFile = open(predictionFilePath, 'a+')'''
                cluster_dict = defaultdict(list)
                cluster_dictPath =defaultdict(list)
                New_coordList = []
                New_categoryList=[]

                neigh = KNeighborsClassifier(n_neighbors=k)
                neigh.fit(coordList,categoryList)

                last=2
                for u in range(1,last+1):
                    #neigh.fit(coordList,categoryList)
                    i = 0
                    for x, y in zip(coordList, categoryList):
                            coordDict={}
                            coordDict["latitude"]=x[0]
                            coordDict["longitude"]=x[1]
                            #print(x)
                            #valIndex = coordList.index(x)
                            New_coordList.append(x)
                            #print("Val index")
                            #print(valIndex)

                            #xFix=coordList[:]
                            #yFix=categoryList[:]

                            #del xFix[valIndex]
                            #del yFix[valIndex]

                            
                            if i%1000==0:
                                print(i, u)
                            i= i+1
                            #x = x.reshape(1,-1)
                            fx = neigh.predict(np.asarray(x).reshape(1,-1))
                            #print(fx)
                            New_categoryList.append(fx[0])
                   
                            #filename
                            fileName=datasetPath+"/"+cnnModelName+"/tagsReduced"+measure+"/lat="+coordDict["latitude"].replace(".",",") +"_long="+ coordDict["longitude"].replace(".",",") +".txt"

                            #x = x.reshape(1,-1)
                            #print(x)
                            #fx = neigh.predict(x)
                            if u == last:
                                cluster_dict[fx[0]].append(coordDict)
                                cluster_dictPath[fx[0]].append(fileName)

                     #reasignar valores a las listas de coordenadas y clusters
                    coordList=New_coordList[:]
                    categoryList=New_categoryList[:]
              

                flName = map.replace(".json","_Cknn_"+str(k)+".json" )

                with open(mapGeneratedPath+"/"+flName, 'w') as fp:  # clusterList_average_braycurtis_0.6_"+"knn"+str(k)+".json
                            json.dump(cluster_dict, fp, sort_keys=True )

                pthFlName = flName.replace("List","ListPaths")
                with open(mapGeneratedPathPth+"/"+pthFlName, 'w') as fp:
                            json.dump(cluster_dictPath, fp, sort_keys=True, indent=4 ) #clusterListPaths_average_braycurtis_0.6_"+"knn"+str(k)+".json

#print( neigh.predict([[38.39523, -0.52098]]))
#print(neigh.predict_proba([[38.39523, -0.52098]]))


#arreglar el nombre del fichero a crear para auq solo se agrege el knn#
