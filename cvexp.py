# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:58:53 2020

@author: igotj
"""


import cv2 as cv;

print(cv.__version__);

videocap = cv.VideoCapture('SwitchTre.mov');

success, image = videocap.read();

fourcc = cv.VideoWriter_fourcc(*'DIVX');
out = cv.VideoWriter('output2.mp4', fourcc, 9.0, (828,1792) );

count = 0;

while success:
    if(count % (9) == 0):
        cv.imwrite("Try 2/frame%d.jpg" % count, image);
        success, image = videocap.read();
        print('Read a new frame: ', success);
        image = cv.flip(image,1);
        out.write(image);
    else:
        videocap.read();
    
    count += 1;
    
videocap.release();
out.release();  
