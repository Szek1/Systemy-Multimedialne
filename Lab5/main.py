import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy
from scipy.interpolate import interp1d


def kwant(signal, bit):
    if bit < 1 or bit > 32:
        print("Zakres wartości musi być od 2 do 32!")
        sys.exit()

    d = 2 ** bit - 1

    if np.issubdtype(signal.dtype, np.floating):
        # co jeżeli float?
        m = -1
        n = 1
    else:
        # co jeżeli integer?
        m = np.iinfo(signal.dtype).min
        n = np.iinfo(signal.dtype).max

    # Przygotowanie
    out_data = signal.astype(float)
    out_data = (out_data - m)
    out_data = out_data/(n-m)
    # kwantyzacja wartości
    out_data = np.round(out_data * d)
    out_data = out_data / d
    # Powrót do oryginalnej przestrzeni
    out_data = (out_data * (n - m)) + m
    out_data = out_data.astype(data.dtype)
    return out_data


def decimation(signal,fs, n):
    N = int(fs/n)
    return signal[::N]


def interpolation(signal, fs, N1, kindLin= True):
    N = signal.shape[0]
    t = np.linspace(0, N/fs, N)
    t1 = np.linspace(0, N/fs, N1)
    if kindLin == True:
        print("Linear")
        metode_lin = interp1d(t,signal)
        y_lin=metode_lin(t1).astype(signal.dtype)
        return y_lin
    else:
        print("cubic")
        metode_nonlin = interp1d(t,signal,kind='cubic')
        y_nonlin = metode_nonlin(t1).astype(signal.dtype)
        return y_nonlin


def plotAudio(Signal, Fs, TimeMargin=[0, 0.1], fsize=2 ** 8):
    x = np.arange(0, Signal.shape[0]) / Fs
    #
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x, Signal)
    plt.ylabel('Amplituda')
    plt.xlabel('Czas w sekundach')
    plt.xlim(TimeMargin[0], TimeMargin[1])

    plt.subplot(2, 1, 2)
    yf = scipy.fftpack.fft(Signal, fsize)
    plt.plot(np.arange(0, Fs / 2, Fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))
    plt.ylabel('dB')
    plt.xlabel('Częstotliwości')
    plt.show()

def plotWidmo(Signal,Fs,n,fsize=2 ** 8):
    plt.subplots_adjust(hspace=0.5)
    # plt.subplot(3,1,1)
    # yf = scipy.fftpack.fft(decimation(Signal,Fs,n), fsize)
    # plt.plot(np.arange(0, Fs / 2, Fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))
    # plt.title(f"Decymacja {n} Hz")

    plt.subplot(2,1,1)
    # plt.subplot(3,1,2)
    yf = scipy.fftpack.fft(interpolation(Signal,Fs,n,True), fsize)
    plt.plot(np.arange(0, Fs / 2, Fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))
    plt.title(f"Interpolacja liniowa {n} Hz")

    plt.subplot(2,1,2)
    # plt.subplot(3,1,3)
    yf = scipy.fftpack.fft(interpolation(Signal, Fs, n, False), fsize)
    plt.plot(np.arange(0, Fs / 2, Fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))
    plt.title(f"Interpolacja nieliniowa {n} Hz")
    plt.show()

data, fs = sf.read('sing_low2.wav', dtype=np.int32)
# data - tablica zawierająca wartości próbek sygnał
# fs - częstotliwość próbkowania -  jak wiele próbek znajduje się w
# jednej sekundzie trwania dźwięku.

# print(f"{np.iinfo(data.dtype).min} \n {np.iinfo(data.dtype).max}")
# print(data)
# print(data.astype(np.float32))
# print(f"{np.issubdtype(data.dtype,np.integer)}\n{np.issubdtype(data.dtype,np.floating)}")

# 60 hz
# plotAudio(kwant(data,4),fs,TimeMargin=[0,0.1])
# plotAudio(kwant(data,8),fs,TimeMargin=[0,0.1])
# plotAudio(kwant(data,16),fs,TimeMargin=[0,0.1])
# plotAudio(kwant(data,24),fs,TimeMargin=[0,0.1])

# 440Hz
# plotAudio(kwant(data,4),fs,TimeMargin=[0,0.05])
# plotAudio(kwant(data,8),fs,TimeMargin=[0,0.05])
# plotAudio(kwant(data,16),fs,TimeMargin=[0,0.05])
# plotAudio(kwant(data,24),fs,TimeMargin=[0,0.05])

# 8000Hz

# plotAudio(kwant(data,4),fs,TimeMargin=[0,0.05])
# plotAudio(kwant(data,8),fs,TimeMargin=[0,0.05])
# plotAudio(kwant(data,16),fs,TimeMargin=[0,0.05])
# plotAudio(kwant(data,24),fs,TimeMargin=[0,0.05])

#
# high1
# high2 low1 low2 medium1 medium 2

xd = np.linspace(-1,1,1000, dtype=np.float32)
plt.plot(kwant(xd,4))
plt.show()
fs1 = [2000, 4000, 8000, 16000,24000,41000]
fs2 =  [16950]
# for i in fs2:
#     plotWidmo(data,fs,i)
bits = [4,8,16,24]

# for i in bits:
#     print(f"{i} bit")
#     sd.play(kwant(data,i))
#     print("end")
#     sd.wait()

# for i in fs1:
#     print(f"{i} Hz")
#     sd.play(decimation(data,fs,i),samplerate=i)
#     print("end")
#     sd.wait()

# for i in fs1:
#     print(f"{i} Hz")
#     sd.play(interpolation(data,fs,i,False),samplerate=i)
#     sd.wait()
#     print("end")
