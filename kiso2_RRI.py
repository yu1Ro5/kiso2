# -*- coding: utf-8 -*-
import numpy as np
import csv

data1, data2, data3, data4 = np.loadtxt("yuro_osignal.txt", unpack=True)
data1 = data1 /1000

flag = False
hako = np.zeros((0,0))

for i in range(len(data1)):
    if data3[i] >= 60000 :
        if flag is True:
            continue
        hako = np.append(hako, data1[i])
        flag = True
        continue
    if data3[i] < 60000 and flag == True:
        if flag is False:
            continue
        hako = np.append(hako, data1[i])
        flag = False
        continue

ri = np.zeros(0)
rri = np.zeros(0)
ihr = np.zeros(0)
for j in range(0, len(hako), 2):
    if j < len(hako)-1:
        ri = np.append(ri, ((hako[j] + hako[j+1])/2))
        
for k in range(1, len(ri)):
    rri = np.append(rri, (ri[k] - ri[k-1]))

for l in range(0, len(rri)):
    ihr = np.append(ihr, (60 / rri[l]))

with open('rri.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(hako.T)
    writer.writerow(ri.T)
    writer.writerow(rri.T)
    writer.writerow(ihr.T)
f.close()
print('FINISH')