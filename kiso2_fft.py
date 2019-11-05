# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import signal
import math
import copy

data1, data2, data3, data4 = np.loadtxt("yuro_osignal.txt", unpack=True)
data1 = data1[45000:len(data1)]
data4 = data4[45000:len(data4)]
avg_data4 = np.mean(data4)
data4 = data4 - avg_data4

def fft_all():
    # データのパラメータ
    N = len(data4)           # サンプル数
    dt = 0.001          # サンプリング間隔
    # f1, f2 = 10, 20    # 周波数
    t = np.arange(0, N*dt, dt) # 時間軸
    freq = np.linspace(0, 1.0/dt, N) # 周波数軸
    '''
    # 信号を生成（周波数10の正弦波+周波数20の正弦波+ランダムノイズ）
    f = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + 0.3 * np.random.randn(N)
    '''
    f = data4
    
    # 高速フーリエ変換
    F = np.fft.fft(f)
    
    # 振幅スペクトルを計算
    Amp = np.abs(F)
    
    # グラフ表示
    plt.figure(figsize=(9,9),dpi=180)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 17
    plt.subplot(121)
    plt.plot(t, f, label='f(n)')
    plt.xlabel("", fontsize=20)
    plt.ylabel("", fontsize=20)
    plt.grid()
    leg = plt.legend(loc=1, fontsize=25)
    leg.get_frame().set_alpha(1)
    plt.subplot(122)
    plt.plot(freq, Amp, label='|F(k)|')
    plt.xlim(0,1.0)
    plt.ylim(0,1000000000)
    plt.xlabel('', fontsize=20)
    plt.ylabel('', fontsize=20)
    plt.grid()
    leg = plt.legend(loc=1, fontsize=25)
    leg.get_frame().set_alpha(1)
    plt.show()
    
def fft_cut():
    for i in range(0,len(data4),100000):
        j = i+100000
        if j > len(data4):
            break
        # データのパラメータ
        N = 100000           # サンプル数
        dt = 0.001          # サンプリング間隔
        # f1, f2 = 10, 20    # 周波数
        t = np.arange(0, N*dt, dt) # 時間軸
        freq = np.linspace(0, 1.0/dt, N) # 周波数軸
        
        f = data4[i:j]
        
        # 高速フーリエ変換
        F = np.fft.fft(f)
        
        # 振幅スペクトルを計算
        Amp = np.abs(F)
        
        # グラフ表示
        plt.figure(figsize=(9,9),dpi=180)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 17
        plt.subplot(121)
        plt.plot(t, f, label='f(n)')
        plt.xlabel("", fontsize=20)
        plt.ylabel("", fontsize=20)
        plt.grid()
        leg = plt.legend(loc=1, fontsize=25)
        leg.get_frame().set_alpha(1)
        plt.subplot(122)
        plt.plot(freq, Amp, label='|F(k)|')
        plt.xlim(0,1.0)
        plt.ylim(0,1000000000)
        plt.xlabel('', fontsize=20)
        plt.ylabel('', fontsize=20)
        plt.grid()
        leg = plt.legend(loc=1, fontsize=25)
        leg.get_frame().set_alpha(1)
        plt.show()

def fft_cut_fin():
    Fg = np.zeros(0)
    for i in range(0,len(data4),100000):
        j = i+100000
        if j > len(data4):
            break
        # データのパラメータ
        N = 100000           # サンプル数
        dt = 0.001          # サンプリング間隔
        # f1, f2 = 10, 20    # 周波数
        t = np.arange(0, N*dt, dt) # 時間軸
        freq = np.linspace(0.1, 1.0, 900) # 周波数軸
        f = data4[i:j]
        
        # 高速フーリエ変換
        F = np.fft.fft(f)
        # 振幅スペクトルを計算
        Amp = np.abs(F)
        Amp = Amp[100:1000]
        
        # グラフ表示
        plt.figure(figsize=(9,9),dpi=180)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 17
        plt.subplot(121)
        plt.plot(t, f, label='f(n)')
        plt.xlabel("", fontsize=20)
        plt.ylabel("", fontsize=20)
        plt.grid()
        leg = plt.legend(loc=1, fontsize=25)
        leg.get_frame().set_alpha(1)
        plt.subplot(122)
        plt.plot(freq, Amp, label='|F(k)|')
        plt.xlim(0,1.0)
        plt.ylim(0,5000000)
        plt.xlabel('', fontsize=20)
        plt.ylabel('', fontsize=20)
        plt.grid()
        leg = plt.legend(loc=1, fontsize=25)
        leg.get_frame().set_alpha(1)
        plt.show()
        
        Fg = np.append(Fg,np.dot(Amp,freq)/np.sum(Amp))
    
    with open('fft.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(Fg)
    f.close()
    print('FINISH')

def filter():
    # 時系列のサンプルデータ作成
    n = len(data4)                         # データ数
    dt = 0.001                       # サンプリング間隔
    # f = 1                           # 周波数
    fn = 1/(2*dt)                   # ナイキスト周波数
    t = np.linspace(1, n, n)*dt-dt
    y = data4
    
    # パラメータ設定
    fp = 1                          # 通過域端周波数[Hz]
    fs = 2                          # 阻止域端周波数[Hz]
    gpass = 1                       # 通過域最大損失量[dB]
    gstop = 40                      # 阻止域最小減衰量[dB]
    # 正規化
    Wp = fp/fn
    Ws = fs/fn
    
    # ローパスフィルタで波形整形
    # バターワースフィルタ
    N, Wn = signal.buttord(Wp, Ws, gpass, gstop)
    b1, a1 = signal.butter(N, Wn, "low")
    y1 = signal.filtfilt(b1, a1, y)
    '''
    # 第一種チェビシェフフィルタ
    N, Wn = signal.cheb1ord(Wp, Ws, gpass, gstop)
    b2, a2 = signal.cheby1(N, gpass, Wn, "low")
    y2 = signal.filtfilt(b2, a2, y)
    
    # 第二種チェビシェフフィルタ
    N, Wn = signal.cheb2ord(Wp, Ws, gpass, gstop)
    b3, a3 = signal.cheby2(N, gstop, Wn, "low")
    y3 = signal.filtfilt(b3, a3, y)
    
    # 楕円フィルタ
    N, Wn = signal.ellipord(Wp, Ws, gpass, gstop)
    b4, a4 = signal.ellip(N, gpass, gstop, Wn, "low")
    y4 = signal.filtfilt(b4, a4, y)
    # ベッセルフィルタ
    N = 4
    b5, a5 = signal.bessel(N, Ws, "low")
    y5 = signal.filtfilt(b5, a5, y)
    
    # FIR フィルタ
    a6 = 1
    numtaps = n
    b6 = signal.firwin(numtaps, Wp, window="hann")
    y6 = signal.lfilter(b6, a6, y)
    delay = (numtaps-1)/2*dt
    '''
    # プロット
    plt.figure()
    plt.plot(t, y, "b")
    plt.plot(t, y1, "r", linewidth=2, label="butter")
    '''
    plt.plot(t, y2, "g", linewidth=2, label="cheby1")
    plt.plot(t, y3, "c", linewidth=2, label="cheby2")
    plt.plot(t, y4, "m", linewidth=2, label="ellip")
    plt.plot(t, y5, "k", linewidth=2, label="bessel")
    plt.plot(t-delay, y6, "y", linewidth=2, label="fir")
    '''
    plt.xlim(0, 100)
    plt.legend(loc="upper right")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

# 波形からノイズを除去する。
# 前半：波形生成
# 　　　２つの周波数のcos波とランダムノイズの足し合わせ
# 後半：ノイズ除去
# 　　　元波形　　　f　元波形のfft　F
# 　　　処理後波形　g　処理後波形のfft　G
#
# 正規化とアンチエリアジングをやめて、
# 共役複素数の項を残すことにしました。18.12.05

def main():
    # データのパラメータ
    N = 40000             # サンプル数
    dt = 1*1e-5           # サンプリング間隔

    # 軸の計算    
    t = np.arange(0, N*dt, dt) # 時間軸
    freq = np.linspace(0, 1.0/dt, N) # 周波数軸

    # サンプル波形のパラメータ
    a = 10                # 交流成分
    fq1, fq2 = 25, 50     # 周波数
    phi1, phi2 = 0, 30    # 位相
    pi = math.pi          # π
    phirad1, phirad2 = phi1*pi/180, phi2*pi/180  # 位相ラジアン
    fc = 200         # カットオフ周波数
    fs = 1 / dt     # サンプリング周波数
    fm = (1/2) * fs # アンチエリアジング周波数
    fc_upper = fs - fc # 上側のカットオフ　fc～fc_upperの部分をカット

    # 時間信号を生成（周波数f1の正弦波+周波数f2の正弦波+ノイズ）
    noise = 0.5 * np.random.randn(N)
    f = a + 1*np.cos(2*np.pi*fq1*t+phirad1) \
          + 1*np.cos(2*np.pi*fq2*t+phirad2) \
          + noise

    # 元波形をfft
    F = np.fft.fft(f)

    # 正規化 + 交流成分2倍
    # F = F/(N/2)
    # F[0] = F[0]/2

    # アンチエリアジング
    # F[(freq > fm)] = 0 + 0j

    # 元波形をコピーする
    G = F.copy()

    # ローパス
    G[((freq > fc)&(freq< fc_upper))] = 0 + 0j

    # 高速逆フーリエ変換
    g = np.fft.ifft(G)

    # 振幅を元に戻す
    # g = g * N
    # 実部の値のみ取り出し
    g = g.real

    # プロット確認
    plt.subplot(221)
    plt.plot(t, f)

    plt.subplot(222)
    plt.plot(freq, F)

    plt.subplot(223)
    plt.plot(t, g)

    plt.subplot(224)
    plt.plot(freq, G)

    plt.show()