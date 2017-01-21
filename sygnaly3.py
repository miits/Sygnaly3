import sys
import scipy.io.wavfile as wv
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
import scipy.signal as sig


def read_file(filename):
    rate, voice_2channels = wv.read(filename)
    voice = voice_2channels
    if len(voice.shape) > 1:
        voice = voice[:, 0]
    return rate, voice


def cut_signal_length(signal, rate, sec):
    length = sec * rate - 1
    cut_signal = signal[0:length]
    return cut_signal


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]


def lowpass(voice, rate, threshold):
    order = 4
    thr = threshold / (rate * 0.5)
    b, a = sig.butter(order, thr, 'low')
    lowpass_voice = sig.filtfilt(b, a, voice)
    return lowpass_voice


rate, voice = read_file("train/003_K.wav")
voice = cut_signal_length(voice, rate, 3)

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
maximum = sig.argrelextrema(auto_correlated, np.greater)
T = abs(maximum[0][0] - maximum[0][1]) / rate
f = 1/T
print(f)
plt.subplot(311)
plt.plot(voice)
plt.subplot(312)
plt.plot(filtered_voice)
plt.subplot(313)
plt.plot(auto_correlated)
plt.show()



