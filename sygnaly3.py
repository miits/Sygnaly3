import sys
import scipy.io.wavfile as wv
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
from scipy.signal import argrelextrema


def read_file(filename):
    rate, voice_2channels = wv.read(filename)
    voice = voice_2channels
    if len(voice.shape) > 1:
        voice = voice[:, 0]
    return rate, voice


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]


def lowpass(voice, rate, threshold):
    spectrum = np.fft.fft(voice)
    spectrum = spectrum.real
    freqs = np.fft.fftfreq(len(voice), 1/rate)
    for i in range(len(voice)):
        if freqs[i] >= threshold:
            spectrum[i] = 0
    lowpass_voice = np.fft.ifft(spectrum)
    return lowpass_voice


rate, voice = read_file("train/002_M.wav")

# ln = len(voice)
# cor = []
#
# for i in range(0, ln, 10):
#     end = ln - 1 - i
#     begin = ln - end
#     cor.append(sts.pearsonr(voice[0:end], voice[begin::])[0])
#
# cor = cor[250::]
#
# i_max_1 = cor.index(max(cor))
# #i_max_1 = i_max_1[0]
#
# diff = 100
# if (i_max_1 - diff) < 0:
#     i_max_2 = cor.index(max(cor[(i_max_1 + diff)::]))
#     #i_max_2 = i_max_2[0] #+ i_max_1 + diff
# elif (i_max_1 + diff) > ln:
#     i_max_2 = np.where(cor == max(cor[0:(i_max_1 - diff)]))
#     #i_max_2 = i_max_2[0]
# else:
#     i_max_21 = cor.index(max(cor[0:(i_max_1 - diff)]))
#     #i_max_21 = i_max_21[0]
#     i_max_22 = cor.index(max(cor[(i_max_1 + diff)::]))
#     #i_max_22 = i_max_22[0] #+ i_max_1 + diff
#     i_max_2 = i_max_21
#     if cor[i_max_22] > cor[i_max_21]:
#         i_max_2 = i_max_22
#
#
# T = abs(i_max_1 - i_max_2) * 10 / rate
# f = 1/T

#print(f)
filtered_voice = lowpass(voice, rate, 500)
auto_correlated = autocorr(filtered_voice)
maximum = argrelextrema(auto_correlated, np.greater)
T = abs(maximum[0][0] - maximum[0][1]) / rate
f = 1/T
sorted = voice[maximum[0]].sort()
print(maximum[0][0:29])
plt.subplot(311)
plt.plot(voice)
plt.subplot(312)
plt.plot(filtered_voice)
plt.subplot(313)
plt.plot(auto_correlated)
plt.show()



