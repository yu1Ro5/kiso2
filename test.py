import numpy as np
from matplotlib import pyplot

data = np.loadtxt('Book1.csv', delimiter=',')
peak = np.zeros((1,2))
flag = True
# for i in range(100000):
for i in range(len(data[:,0])):
    if(data[i][1] > 50000 and data[i][1]-data[i-1][1] < 0 and flag == True):
        peak = np.append(peak,np.array([[data[i][1],data[i][2]]]), axis=0)
        flag = False
    elif(data[i][1] > 50000 and data[i][1]-data[i-1][1] > 0 and flag == False):
        flag = True

peak_all = np.delete(peak, 0, 0)
a = len(peak_all[:,0])
print(a)
rri = np.zeros((1,2))
for i in range(len(peak_all[:,0])-1):
    rri = np.append(rri,np.array([[peak_all[i+1][0],peak_all[i+1][0]-peak_all[i][0]]]), axis=0)

rri_all = np.delete(rri, 0, 0)
# print(rri_all)
freq = np.linspace(0, 1/1.1, len(rri_all[:,0]))
F = np.fft.fft(rri_all[:,1])
Amp = np.abs(F)
# print(freq)
pyplot.plot(freq, Amp)
pyplot.xlim(0,0.5)
pyplot.ylim(0,10)
# pyplot.plot(rri_all[:,0], rri_all[:,1])
pyplot.show()