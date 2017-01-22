import scipy.io.wavfile as wv
import numpy as np
import re
import os
import scipy.signal as sig
from sklearn import linear_model, datasets


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
    return result[result.size / 2:]


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

    # voice = cut_signal_length(voice, rate, 1)


    filtered_voice = lowpass(voice, rate, 1000)
    auto_correlated = autocorr(filtered_voice)
    maximum = sig.argrelextrema(auto_correlated, np.greater)
    sorted_maximum = sorted(maximum[0], key=get_key, reverse=True)
    # T = abs(sorted_maximum[0] - sorted_maximum[1]) / rate
    # T = abs(maximum[0][0] - maximum[0][1]) / rate
    diffs = np.diff(maximum[0])
    T = np.average(diffs) / rate
    f = 1 / T
    if (re.match("\d\d\d_M.wav", filename)):
        Y_set.append("M")
    else:
        Y_set.append("K")
    g = []
    g.append(f)
    f_set.append(g)
    print(f, ' - ', filename)


def main():
    global f_set
    global Y_set
    Y_set = []
    f_set = []
    files = []
    # files.append('010_M.wav')
    # files.append('011_M.wav')
    # files.append('013_M.wav')
    # files.append('017_M.wav')
    #
    # files.append('014_K.wav')
    # files.append('015_K.wav')
    # files.append('016_K.wav')
    # files.append('018_K.wav')

    # for file in files:
    #     gender(file)
    for filename in os.listdir('train'):
        gender(filename)

    logistic = linear_model.LogisticRegression(C=1e5)

    logreg = logistic.fit(f_set, Y_set)
    i = 0
    predict = logreg.predict(f_set)
    miss_count = 0
    for file in files:
        p = predict[i]
        if p not in file:
            miss_count +=1
        print("plik - ", file, " predict - ", p)
        i +=1
    print("miss = ", miss_count)

main()
