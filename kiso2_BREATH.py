# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 23:33:34 2019

@author: yuuro
"""
import numpy as np
import csv
import matplotlib.pyplot as plt

data1, data2, data3, data4 = np.loadtxt("yuro_osignal.txt", unpack=True)
data1 = data1 /1000

# 描画範囲の指定
# x = np.arange(x軸の最小値, x軸の最大値, 刻み)
x = data1

# 計算式
y = data3

# グラフ表示
plt.figure(figsize=(9,9),dpi=180)

# 横軸の変数。縦軸の変数。
plt.plot(x, y)

plt.xlim([0,20])

# 描画実行
plt.show()