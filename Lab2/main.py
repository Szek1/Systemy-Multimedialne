import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.fftpack


# Część druga

# data, fs = sf.read('sound1.wav', dtype='float32')


# print(f"Data Dtype = {data.dtype}" )
# # wartości próbek sygnału
# # i kanały lewy i prawy
# print(f"data shape = {data.shape}")
# print(f"fs = {fs}" )
#  częstotliwość próbkowania,
#  czyli jak bardzo sygnał rzeczywisty został spróbkowany,
#  czyli jak wiele próbek znajduje się w jednej sekundzie trwania dźwięku.


# Zad1
# data, fs = sf.read('sound1.wav', dtype='float32')


# newDataLeftChannel = data[:,0]
# newDataRightChannel = data[:,1]
# mono = (newDataLeftChannel + newDataRightChannel)/2
#
#
# sf.write('sound_L.wav',newDataLeftChannel,fs)
# sf.write('sound_R.wav',newDataRightChannel,fs)
# sf.write('sound_mix.wav',mono,fs)

# Część trzecia
# data, fs = sf.read('sound1.wav', dtype='float32')
# x = np.arange(0,data.shape[0])/fs
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(x,data[:,0])
#
# plt.subplot(2,1,2)
# plt.plot(x,data[:,1])
# plt.show()



# data, fs = sf.read('sin_440Hz.wav', dtype=np.float32)

# 1 przykład

# x = np.arange(0,data.shape[0])/fs
#
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(x,data)
#
# plt.subplot(2,1,2)
# yf = scipy.fftpack.fft(data)
# plt.plot(np.arange(0,fs,1.0*fs/(yf.size)),np.abs(yf))
# plt.show()


# 2 przykład

# fsize=2**8

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(np.arange(0,data.shape[0])/fs,data)
#
# plt.subplot(2,1,2)
# yf = scipy.fftpack.fft(data,fsize)
# plt.plot(np.arange(0,fs,fs/fsize),np.abs(yf))
# plt.show()

# 3 przykład

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(np.arange(0,data.shape[0])/fs,data)
# plt.subplot(2,1,2)
# yf = scipy.fftpack.fft(data,fsize)
# plt.plot(np.arange(0,fs/2,fs/fsize),np.abs(yf[:fsize//2]))
# plt.show()
# data, fs = sf.read('sin_440Hz.wav', dtype=np.float32)
# fsize=2**8
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(np.arange(0,data.shape[0])/fs,data)
# plt.subplot(2,1,2)
# yf = scipy.fftpack.fft(data,fsize)
# plt.plot(np.arange(0,fs/2,fs/fsize),20*np.log10( np.abs(yf[:fsize//2])))
# plt.show()


def plotAudio(Signal,Fs,TimeMargin=[0,0.02],fsize=2**8):
    x = np.arange(0,Signal.shape[0])/Fs

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(x,Signal)
    plt.ylabel('Amplituda')
    plt.xlabel('Czas w sekundach')
    plt.xlim(TimeMargin[0],TimeMargin[1])

    plt.subplot(2,1,2)
    yf = scipy.fftpack.fft(Signal, fsize)
    plt.plot(np.arange(0, Fs / 2, Fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))
    plt.ylabel('dB')
    plt.xlabel('Częstotliwości')
    plt.show()

data, fs = sf.read('sin_440Hz.wav', dtype=np.int32)

plotAudio(data,fs,[0,1])
