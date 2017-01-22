import scipy.io.wavfile as wv
import numpy as np
import re
import sys
import scipy.signal as sig
from sklearn import linear_model, datasets
import os
import pickle


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
    f = get_base_freq(voice, rate)
    if (re.match("\d\d\d_M.wav", filename)):
        Y_set.append("M")
    else:
        Y_set.append("K")
    g = []
    g.append(f)
    f_set.append(g)
    print(f, ' - ', filename)


def get_base_freq(voice, rate):
    filtered_voice = lowpass(voice, rate, 1000)
    auto_correlated = autocorr(filtered_voice)
    maximum = sig.argrelextrema(auto_correlated, np.greater)
    diffs = np.diff(maximum[0])
    T = np.average(diffs) / rate
    f = 1 / T
    return f


def train():
    global f_set
    global Y_set
    Y_set = []
    f_set = []
    files = os.listdir('train')
    for filename in files:
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

    pickle.dump(logreg, open("logreg.p", "wb"))


def main():
    logreg = pickle.load(open("logreg.p", "rb"))
    filename = sys.argv[1]
    global voice
    rate, voice = read_file("train/" + filename)
    f = get_base_freq(voice, rate)
    predicted_gender = logreg.predict(f)[0]
    print(predicted_gender)

main()