# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:15:34 2020

@author: igotj
"""

#import module statements
import os
import random

filelist = list()
   
for root, dirs, files in os.walk("Tricks", topdown=True):
    for name in files:
        if name.endswith(".mov"):
            if(filelist == None):
                filelist=list()
            path = os.path.join(root,name).replace("\\",'/')
            path = path.strip("Tricks/")
            filelist.append(path)
    
random.shuffle(filelist)
    
trainlist = len(filelist)*0.80
trainlist = int(trainlist)

with open("trainlist02.txt","w") as f:
    for i in range(0,trainlist):
        f.write(filelist[i]+"\n")
        
with open("testlist02.txt","w") as f:
    for j in range(trainlist+1, len(filelist)):
        f.write(filelist[j]+"\n")       