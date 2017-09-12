#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 16:40:21 2017

@author: lenkakt
"""
import os, sys, getopt
import gravi

import numpy as np

strMainHelp = 'Usage: \n ' + \
    'get_data.py -o <OutputFolder> -s <DataSize> -x <XLength> -y <YLength> -a <XStep> -b <YStep> -n <AddNoise> -l <NoiseLevel> -d <DataSeparator> -c <PrintComment> -X <XLimit>'+ \
    '\n Example: get_data.py --DataSize=1000'
strShortHelp = 'get_data.py -o <OutputFolder> -s <DataSize> -x <XLength> -y <YLength> -a <XStep> -b <YStep> -n <AddNoise> -l <NoiseLevel> -d <DataSeparator> -c <PrintComment> -X <XLimit>' 


def main(argv):
#---------------------------------------------------
#  Configuration / parameters to be set 
#
# 
    OutputFolder = './'
    DataSize = 1
    XLength = 100 #X lenght of the area in meters
    YLength = 100 #Y length of the area in meters 
    XStep = 1     #The size of sampling step in meters
    YStep = 1     #The size of sampling step in meters
    AddNoise = False
    NoiseLevel = 0.1 #The Noise level in percents
    AnomalyTypes = [0,1,2]   #The index of an anomaly type defined in Anomalies. Should be moved to parameters from command line      
    Densities = [1e6,1e4,1e4]    #The list of densities for each of anomaly types
    MaxDepth = 50
    MaxRadius = 20
    DataSeparator = ','
    PrintComment = False
    XLimit = 0
#----------------------------------------------------------------------------
#
# List of defined anomalies:
#
    Anomalies = ["Sphere - density","Horizontal cylinder","Vertical cylinder","Random"]
    #AnomaliesFcn = ["sphere_density","cylinder_horizontal","cylinder_vertical","random_data"]
    
    try:
        opts, args = getopt.getopt(argv,"ho:s:x:y:a:b:nl:d:cX:",["OutputFolder=","DataSize=","XLength=","YLength=","XStep=","YStep=","AddNoise","NoiseLevel=","DataSeparator=","PrintComment","XLimit="])
    except getopt.GetoptError:
        print strShortHelp
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print strMainHelp
            sys.exit()
        elif opt in ("-o", "--OutputFolder"):
            OutputFolder = arg
            if not os.path.exists(OutputFolder):
                print "Creating the output directory " + OutputFolder
                os.makedirs(OutputFolder)            
 
        elif opt in ("-s", "--DataSize"):
            DataSize = int(arg)
        elif opt in ("-x","--XLength"):
            XLength = int(arg)
        elif opt in ("-y","--YLength"):
            YLength = int(arg)
        elif opt in ("-a","--XStep"):
            XStep = int(arg)
        elif opt in ("-b","--YStep"):
            YStep = int(arg)
        elif opt in ("-n","--AddNoise"):
            AddNoise = True
        elif opt in ("-l","--NoiseLevel"):
            NoiseLevel = float(arg)
        elif opt in ("-d","--DataSeparator"):            
            DataSeparator = str(arg)
        elif opt in ("-c","--PrintComment"):
            PrintComment = True
        elif opt in ("-X","--XLimit"):
            XLimit = int(arg)
    
            
#------------------------------------------------------------------------
# 
#Create a grid
#
#
    XPoints = np.arange(0,XLength,XStep)
    YPoints = np.arange(0,YLength,YStep)
    XScale, YScale = np.meshgrid(XPoints,YPoints)
    
#--------------------------------------------------------------------------
# Set random parameters
#    
    Depths = np.random.randint(1,MaxDepth,(DataSize,1))
    Radiuses = np.random.randint(1,MaxRadius,(DataSize,1))

#-----------------------------------------------------------------------
# Set print options
#    
    np.set_printoptions(threshold=np.inf, linewidth=np.inf) 

# Main part:
# Points: always have [X1,Y1] or [X1,Y1,X2,Y2]
# Depths always depth from 1 - 20
# Radius always Radius from 1 - 20

    for Anomaly in AnomalyTypes:
        if Anomaly==0: #Sphere             
            Points = np.random.randint(1+XLimit,XLength-XLimit,(DataSize,2))
            
            for i in range(0,DataSize):
                YPos = Points[i,0]
                XPos = Points[i,1]
                ZPos = Depths[i,0]
                Radius = Radiuses[i,0]
                
                if (Radius > ZPos):
                    store = Radius
                    Radius = ZPos
                    ZPos = store
                
                Density = Densities[Anomaly]
                DescriptionString = Anomalies[Anomaly] + " XPos: "+str(XPos)+" YPos: "+str(YPos)+ " ZPos: "+str(ZPos)+" Density: "+str(Density)+" Radius: "+str(Radius)+" XLength: "+str(XLength)+" XStep: "+str(XStep)+" YLength: "+str(YLength)+" YStep: "+str(YStep)
                FileName = Anomalies[Anomaly]+"_"+str(XPos)+"_"+str(YPos)+"_"+str(ZPos)+"_"+str(Density)+"_"+str(Radius)+"_"+str(XLength)+"_"+str(XStep)+"_"+str(YLength)+"_"+str(YStep)+".txt"
                AnomalyMatrix = gravi.sphere_mass(XPos,YPos,ZPos,Density,Radius,XScale,YScale)
                
                if AddNoise:
                    DataScale = NoiseLevel*np.amax(AnomalyMatrix)  
                    NoiseMatrix = gravi.random_data(DataScale,XLength,YLength)
                    AnomalyMatrix = AnomalyMatrix + NoiseMatrix
                
                FilePath = OutputFolder + "/" + FileName
                
                if os.path.exists(FilePath):
                    try:
                        os.remove(FilePath)
                    except OSError, e:
                        print ("Error: %s - %s." % (e.filename,e.strerror))    
 
                with open(FilePath, 'w') as f:
                    if PrintComment:
                        f.writelines('#'+DescriptionString+"\n")
                    f.writelines(np.array2string(AnomalyMatrix,separator=DataSeparator))
                    f.close()
                if (i % 100 == 0):
                   print Anomalies[Anomaly]
                   print i

                
        elif Anomaly == 1: #HCylinder
            Points = np.random.randint(1,XLength,(DataSize,4))
            
            for i in range(0,DataSize):
               YPos = Points[i,0]
               XPos = Points[i,1]
               YPos2 = Points[i,2]
               XPos2 = Points[i,3]
               ZPos = Depths[i,0]
               Radius = Radiuses[i,0]
               
               if (Radius > ZPos):
                    store = Radius
                    Radius = ZPos
                    ZPos = store
               
               Density = Densities[Anomaly]
               DescriptionString = Anomalies[Anomaly] + " XPos: "+str(XPos)+" YPos: "+str(YPos)+ " XPos2: "+str(XPos2)+" YPos2: "+str(YPos2)+" ZPos: "+str(ZPos)+" Density: "+str(Density)+" Radius: "+str(Radius)+" XLength: "+str(XLength)+" XStep: "+str(XStep)+" YLength: "+str(YLength)+" YStep: "+str(YStep)
               FileName = Anomalies[Anomaly]+"_"+str(XPos)+"_"+str(YPos)+"_"+str(XPos2)+"_"+str(YPos2)+"_"+str(ZPos)+"_"+str(Density)+"_"+str(Radius)+"_"+str(XLength)+"_"+str(XStep)+"_"+str(YLength)+"_"+str(YStep)+".txt"
               AnomalyMatrix = gravi.cylinder_horizontal(XPos,YPos,XPos2,YPos2,ZPos,Density,Radius,XScale,YScale)
               
               if AddNoise:
                    DataScale = NoiseLevel*np.amax(AnomalyMatrix)  
                    NoiseMatrix = gravi.random_data(DataScale,XLength,YLength)
                    AnomalyMatrix = AnomalyMatrix + NoiseMatrix
                
               FilePath = OutputFolder + "/" + FileName
                
               if os.path.exists(FilePath):
                   try:
                       os.remove(FilePath)
                   except OSError, e:
                       print ("Error: %s - %s." % (e.filename,e.strerror))    
 
               with open(FilePath, 'w') as f:
                   if PrintComment:
                       f.writelines('#'+DescriptionString+"\n")
                   f.writelines(np.array2string(AnomalyMatrix,separator=DataSeparator))
                   f.close()
               if (i % 100 == 0):
                   print Anomalies[Anomaly]
                   print i

        elif Anomaly == 2: #VCylinder
            Points = np.random.randint(1,XLength,(DataSize,2))
                        
            for i in range(0,DataSize):
                YPos = Points[i,0]
                XPos = Points[i,1]
                ZPos = Depths[i,0]
                Radius = Radiuses[i,0]
                
                if (Radius > ZPos):
                    store = Radius
                    Radius = ZPos
                    ZPos = store
                    
                Density = Densities[Anomaly]
                DescriptionString = Anomalies[Anomaly] + " XPos: "+str(XPos)+" YPos: "+str(YPos)+ " ZPos: "+str(ZPos)+" Density: "+str(Density)+" Radius: "+str(Radius)+" XLength: "+str(XLength)+" XStep: "+str(XStep)+" YLength: "+str(YLength)+" YStep: "+str(YStep)
                FileName = Anomalies[Anomaly]+"_"+str(XPos)+"_"+str(YPos)+"_"+str(XPos2)+"_"+str(YPos2)+"_"+str(ZPos)+"_"+str(Density)+"_"+str(Radius)+"_"+str(XLength)+"_"+str(XStep)+"_"+str(YLength)+"_"+str(YStep)+".txt"
                AnomalyMatrix = gravi.cylinder_vertical(XPos,YPos,ZPos,Density,Radius,XScale,YScale)
                
                if AddNoise:
                    DataScale = NoiseLevel*np.amax(AnomalyMatrix)  
                    NoiseMatrix = gravi.random_data(DataScale,XLength,YLength)
                    AnomalyMatrix = AnomalyMatrix + NoiseMatrix
                
                FilePath = OutputFolder + "/" + FileName
                
                if os.path.exists(FilePath):
                    try:
                        os.remove(FilePath)
                    except OSError, e:
                        print ("Error: %s - %s." % (e.filename,e.strerror))    
 
                with open(FilePath, 'w') as f:
                    if PrintComment:
                        f.writelines('#'+DescriptionString+"\n")                    
                    f.writelines(np.array2string(AnomalyMatrix,separator=DataSeparator))
                    f.close()
                if (i % 100 == 0):
                    print Anomalies[Anomaly]
                    print i

                
        else:
            print 'Unknown anomaly type index'
        


    
    

   
if __name__ == "__main__":
   main(sys.argv[1:])
