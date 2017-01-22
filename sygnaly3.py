import os
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


def get_key(i):
    return voice[i]


def gender(filename):
    global voice
    rate, voice = read_file("train/" + filename)

    #voice = cut_signal_length(voice, rate, 1)


    filtered_voice = lowpass(voice, rate, 1000)
    auto_correlated = autocorr(filtered_voice)
    maximum = sig.argrelextrema(auto_correlated, np.greater)
    sorted_maximum = sorted(maximum[0], key=get_key, reverse=True)
    #T = abs(sorted_maximum[0] - sorted_maximum[1]) / rate
    #T = abs(maximum[0][0] - maximum[0][1]) / rate
    diffs = np.diff(maximum[0])
    T = np.average(diffs) / rate
    f = 1/T
    print(f, ' - ', filename)
    # plt.subplot(311)
    # plt.plot(voice)
    # plt.subplot(312)
    # plt.plot(filtered_voice)
    # plt.subplot(313)
    # plt.plot(auto_correlated)
    # plt.show()

def main():
    # files = []
    # files.append('010_M.wav')
    # files.append('011_M.wav')
    # files.append('013_M.wav')
    # files.append('017_M.wav')
    #
    # files.append('014_K.wav')
    # files.append('015_K.wav')
    # files.append('016_K.wav')
    # files.append('018_K.wav')
    #
    # for file in files:
    #     gender(file)
    for filename in os.listdir('train'):
        gender(filename)

main()