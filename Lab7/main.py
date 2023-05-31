import numpy as np
import matplotlib.pyplot as plt
import sys
import soundfile as sf
import sounddevice as sd


# R = np.random.rand(5, 5)
# A = np.zeros(R.shape)
# B = np.zeros(R.shape)
# C = np.zeros(R.shape)
#
# idx = R < 0.25
# A[idx] = 1  # <-
# B[idx] += 0.25  # <-
# C[idx] = 2 * R[idx] - 0.25  # <-
#
# # C[np.logical_not(idx)]=4*R[np.logical_not(idx)]-0.5 # <-
# print(R, "\n")
# print(A, "\n")
# print(B, "\n")
# print(C, "\n")
# print(R[idx])

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
    out_data = (out_data - m) / (n - m)
    # kwantyzacja wartości
    out_data = np.round(out_data * d)
    out_data = out_data / d
    # Powrót do oryginalnej przestrzeni
    out_data = (out_data * (n - m)) + m
    out_data = out_data.astype(signal.dtype)
    return out_data


def compress_a_law(data, A=87.6):
    signal = np.zeros(data.shape)
    sign = np.sign(data)
    idx = np.where(np.abs(data) < (1 / A))
    signal[idx] = sign[idx] * ((A * np.abs(data[idx])) / (1 + np.log(A)))
    idx = np.where((1 / A <= np.abs(data)))
    signal[idx] = sign[idx] * ((1 + np.log(A * np.abs(data[idx]))) / (1 + np.log(A)))
    return signal


def decompress_a_law(data, A=87.6):
    signal = np.zeros(data.shape)
    sign = np.sign(data)
    idx = np.where(np.abs(data) < 1 / (1 + np.log(A)))

    signal[idx] = (np.abs(data[idx]) * (1 + np.log(A))) / A

    idx2 = np.where((1 / (1 + np.log(A)) <= np.abs(data)) & (np.abs(data) <= 1))

    signal[idx2] = (np.exp(np.abs(data[idx2]) * (1 + np.log(A)) - 1)) / A
    return signal * sign


def compress_mu_law(data, mu=255):
    signal = np.zeros(data.shape)
    idx = np.where((-1 <= data) & (data <= 1))
    signal[idx] = np.sign(data[idx]) * (np.log(1 + mu * np.abs(data[idx]))) / (np.log(1 + mu))
    return signal


def decompress_mu_law(data, mu=255):
    signal = np.zeros(data.shape)
    idx = np.where((-1 <= data) & (data <= 1))
    signal[idx] = np.sign(data[idx]) * 1 / mu * (np.power(1 + mu, np.abs(data[idx])) - 1)
    return signal


def DPCM_compress(x, bit):
    y = np.zeros(x.shape)
    e = 0
    for i in range(0, x.shape[0]):
        y[i] = kwant(x[i] - e, bit)
        e += y[i]
    return y


def DPCM_decompress(signal):
    out_sig = signal.copy()
    for x in range(1, signal.shape[0]):
        out_sig[x] = signal[x] + out_sig[x - 1]
    return out_sig


def DPCM_predict_compress(x, bit, predictor, n):
    y = np.zeros(x.shape)
    xp = np.zeros(x.shape)
    e = 0
    for i in range(1, x.shape[0]):
        y[i] = kwant(x[i] - e, bit)
        xp[i] = y[i] + e
        idx = (np.arange(i - n, i, 1, dtype=int) + 1)
        idx = np.delete(idx, idx < 0)
        e = predictor(xp[idx])
    return y


def DPCM_predict_decompress(signal, predictor,n):
    out_sig = signal.copy()
    xp = signal.copy()
    y = signal.copy()
    e = 0
    for x in range(1, signal.shape[0]):
        out_sig[x] = y[x] + e
        xp[x] = y[x] + e
        idx = (np.arange(x - n, x, 1, dtype=int) + 1)
        idx = np.delete(idx, idx < 0)
        e = predictor(xp[idx])
    return out_sig


def alaw_all(sig, bit):
    comp = compress_a_law(sig)
    quan = kwant(comp, bit)
    return decompress_a_law(quan)


def mu_law_all(sig, bit):
    comp = compress_mu_law(sig)
    quan = kwant(comp, bit)
    return decompress_mu_law(quan)


def DPCM_all(sig, bit):
    comp = DPCM_compress(sig, bit)
    return DPCM_decompress(comp)

def DPCM_predict_all(sig, bit, predicator, n):
    comp = DPCM_predict_compress(sig,bit,predicator,n)
    return DPCM_predict_decompress(comp,predicator,n)

x = np.linspace(-1, 1, 1000)
y = 0.9 * np.sin(np.pi * x * 4)

# plt.subplot(3,1,1)
# plt.plot(x,y)
# plt.title("Oryginalny")
#
# plt.subplot(3,1,2)
# plt.title("A-Law 8 bitów")
# plt.plot(x,alaw_all(y,8))
#
# plt.subplot(3,1,3)
# plt.plot(x,mu_law_all(y,8))
# plt.title("μ-law 8 bitów")
# plt.show()
#
# plt.subplot(3,1,1)
# plt.plot(x,y)
# plt.title("Oryginalny")
#
# plt.subplot(3,1,2)
# plt.title("DPCM 8 bitów")
# plt.plot(x,DPCM_all(y,8))
#
# plt.subplot(3,1,3)
# plt.title("DPCM z predykcją 8 bitów")
# plt.plot(x,DPCM_predict_all(y,8,np.median,4))
# plt.show()

# 2.1 Badanie jakości dźwięku po działaniu domyślnym

data, f = sf.read('sing_high2.wav', dtype='float32')

bits = 2
n = 3
dane = [alaw_all(data, bits), mu_law_all(data, bits), DPCM_all(data, bits),DPCM_predict_all(data,bits,np.mean,n)]
strings = ["A-Law", "Mu-Law", "DPCM", "DPCM predict"]

print("\tLiczba bitów: ", bits)
for i, z in zip(dane, strings):
    print(z)
    sd.play(i, f)
    sd.wait()
