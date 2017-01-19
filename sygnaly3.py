import sys
import scipy.io.wavfile as wv
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts


filename = "train/002_M.wav"
rate, voice_2ch = wv.read(filename)
voice = np.asarray(voice_2ch);
voice = voice[:, 0]

ln = len(voice)
cor = np.zeros(ln)


for i in range(0, ln, 10):
    end = ln - 1 - i
    begin = ln - end
    cor[i] = (sts.pearsonr(voice[0:end], voice[begin::])[0])

cor = cor[250::]

i_max_1 = np.where(cor == max(cor))
i_max_1 = i_max_1[0]

diff = 100
if (i_max_1 - diff) < 0:
    i_max_2 = np.where(cor == max(cor[(i_max_1 + diff)::]))
    i_max_2 = i_max_2[0] #+ i_max_1 + diff
elif (i_max_1 + diff) > ln:
    i_max_2 = np.where(cor == max(cor[0:(i_max_1 - diff)]))
    i_max_2 = i_max_2[0]
else:
    i_max_21 = np.where(cor == max(cor[0:(i_max_1 - diff)]))
    i_max_21 = i_max_21[0]
    i_max_22 = np.where(cor == max(cor[(i_max_1 + diff)::]))
    i_max_22 = i_max_22[0] #+ i_max_1 + diff
    i_max_2 = i_max_21
    if cor[i_max_22] > cor[i_max_21]:
        i_max_2 = i_max_22


T = abs(i_max_1 - i_max_2)*10 / rate
f = 1/T

print(f)
plt.plot(cor)
plt.show()



