import numpy as np
from matplotlib import pyplot

data = np.loadtxt('yuro_osignal.txt', unpack=True)
data[1,:] = data[0,:] / 1000
data = data.T
print(data[:,1])

peak_b = np.zeros((1,2))
flag2 = True
count = 0

for i in range(len(data[:,0])-500):
    if(data[i+500][3] > 33000 and data[i+500+500][3]-data[i+500-500][3] < 0 and flag2 == True and count>1000):
        count = 0
        peak_b = np.append(peak_b, np.array([[data[i+500][1],data[i+500][3]]]), axis=0)
        flag2 = False
        count += 1
    elif(data[i+500][3] > 33000 and data[i+500+500][3]-data[i+500-500][3] > 0 and flag2 == False):
        flag2 = True
        count += 1
    else:
        count += 1


peak_b_all = np.delete(peak_b, 0, 0)

rri = np.zeros((1,2))
for i in range(len(peak_b_all[:,0])-1):
    rri = np.append(rri,np.array([[peak_b_all[i+1][0],peak_b_all[i+1][0]-peak_b_all[i][0]]]), axis=0)

rri_all = np.delete(rri, 0, 0)
a = len(rri_all[:,0])
print(a)

ave = np.average(rri_all[:,1])
print(ave)

std = np.zeros((1,2))
div_sum = 0
count2 = 1
count_num = 0
for i in range(len(rri_all[:,0])):
    if(rri_all[i][0] > 100 * count2):
        div_sum = div_sum + (rri_all[i][1]-ave)**2
        div = div_sum/count_num
        std = np.append(std, np.array([[rri_all[i][0],pow(div,0.5)]]), axis=0)
        div_sum = 0
        count2 += 1
        count_num = 0
    else:
        div_sum = div_sum + (rri_all[i][1]-ave)**2
        count_num += 1
        
std_all = np.delete(std, 0, 0)
np.savetxt('breath.csv',std_all,delimiter=',')

# pyplot.plot(rri_all[:,0], rri_all[:,1])
pyplot.plot(std_all[:,0], std_all[:,1],marker = 'o')
pyplot.show()