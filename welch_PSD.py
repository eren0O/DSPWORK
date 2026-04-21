import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

N = int(1e6)
fs = 1e6
f = 250e3
#welch's method: 
#1. split into segments
#2. window each segment
#3. calculate fft squared 
#4. average out each segment

t = np.arange(N)/fs
awgn = np.sqrt(5)*(np.random.randn(N)+1j*np.random.randn(N))
signal = np.exp(1j*2*np.pi*f*t)
signal = signal+np.exp(1j*2*np.pi*(f+6e3)*t)+awgn # 1e6/256= 3900hz

freq_w, welchpower1 = welch(signal, fs, window="hamming", nperseg=256)#hamming window multiplier x1.81

freq_w1 , welchpower2 = welch(signal, fs, window="hamming", nperseg=4096)

freq_w2 , welchpower3 = welch(signal, fs, window="hamming", nperseg=65536)

plt.figure(figsize=(10,8))
plt.subplot(3,1,1)
plt.plot(freq_w, 
10*np.log10(welchpower1), color="green",label="nperseg=256")
plt.xlabel("Frequency")
plt.ylabel("PSD in dB")
plt.legend()
plt.subplot(3,1,2)
plt.plot(freq_w1, 
10*np.log10(welchpower2), color="blue",label="nperseg=4096")
plt.xlabel("Frequency")
plt.ylabel("PSD in dB")
plt.legend()
plt.subplot(3,1,3)
plt.plot(freq_w2, 
10*np.log10(welchpower3), color="red",label="nperseg=65536")
plt.xlabel("Frequency")
plt.ylabel("PSD in dB")
plt.legend()
plt.show()
#n = fs*duration 
#freq bin = frequency resolution = amount of ferquency represented by a single sample = Δf = fs/N
#nperseg increases => bin width decreases. If I want to catch a jammer that uses narrowband 125hz
#i need lower frequency bin than 125hz to make sure that i dont take him as a hump with the other ones.  
# windowings increase the main lobes bandwith, with a tradeoff by reducing side lobes 