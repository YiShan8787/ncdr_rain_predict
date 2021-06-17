# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:24:10 2021

@author: user
"""

import os

station_path = "E:\\tech\\ncdr\\ncdr_rain_predict\\data\\station_data\\2016\\05\\20160501"

for file in os.listdir(station_path):
    #print(file)
    file_name = station_path + "\\" + file
    f = open(file_name)
    stations = []
    cnt = 0
    for line in f.readlines():
        test = line.split()
        cnt = cnt + 1
        if test[0] not in stations:
            stations.append(test[0])
