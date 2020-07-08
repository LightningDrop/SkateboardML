# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 14:15:34 2020

@author: igotj
"""

#import module statements
import os

#start of loop
with open("trainlist02.txt", "w") as file:
   
    for root, dirs, files in os.walk("Tricks", topdown=True):
        for name in files:
            file.write(os.path.join(root,name).replace("\\","/") + '\n')

temp = "None"    
with open("trainlist02.txt", "r+") as f:
    read_data = f.read()
    f.seek(0)
    f.truncate(0)
    read_data = read_data.replace("Tricks/", "")
    temp = read_data
    f.write(read_data)

with open("testlist02.txt", "w") as f:
    f.write(temp)
    
    
    
    
            
        

        

