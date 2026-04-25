import cupy as cp
from cupyx.scipy.signal import fftconvolve, welch, firwin
import matplotlib.pyplot as plt
import json #will create json metadata + real data when decimation ends

fs = 2.4e6
freq_offset = 400e3 #there is 400e3 offset in bw
target_bw = 300e3 #meaning -150e3 +150e3, but its between 250e3 and 550e3
decimation = 4
#should output: decimated output #so for decimation, need to apply lpf first to prevent aliasing
#should output: PSD before and after
N = int(2.4e6)

data = cp.fromfile("/home/eren/Desktop/captured_signal.dat",dtype=cp.complex64)[:N]#dotn overload ram

numtaps = 201
cutoff = 170e3 #realistic cutoff, make sure it contains all
taps = firwin(numtaps, cutoff, fs=fs)

t = cp.arange(len(data))/fs

shifted_signal = cp.exp(1j*2*cp.pi*-freq_offset*t)*data 

filteredsignal = fftconvolve(shifted_signal,taps ,mode="same")

decimatedsignal = filteredsignal[::decimation] #catch every decimation'th sample, new fs=fs/decimation

fs_new = fs/decimation

freqfilt, filteredsignalPSD = welch(decimatedsignal, fs_new, window="hamming",detrend=False, return_onesided=False)
freq, signalPSD = welch(data, fs, window="hamming", detrend=False, return_onesided=False)

filteredsignalPSD = cp.fft.fftshift(filteredsignalPSD)
freqfilt = cp.fft.fftshift(freqfilt)
freq = cp.fft.fftshift(freq)
signalPSD = cp.fft.fftshift(signalPSD)

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.title("PSD before")
signalPSD = 10*cp.log10(signalPSD+1e-10)
plt.plot(freq.get(), signalPSD.get())
plt.subplot(2,1,2)
plt.title("PSD After filter")
filteredsignalPSD = 10*cp.log10(filteredsignalPSD+1e-10)
plt.plot(freqfilt.get(), filteredsignalPSD.get())
plt.show()

#generate new data with new metadata
decimatedsignal.astype(cp.complex64).tofile("/home/eren/Desktop/captured_signal_processed.dat")

original_center_freq =  926.6e6 #0.5PPM TXCO, might have drifted slighlty, generate for metadata prep

metadata = {
    "original_sample_rate_hz": fs,
    "new_sample_rate_hz": fs_new,
    "original_center_frequency_hz": original_center_freq,
    "target_offset_hz": freq_offset,
    "effective_center_frequency_hz": original_center_freq + freq_offset,
    "target_bandwidth_hz": target_bw,
    "filter_cutoff_hz": cutoff,
    "decimation": decimation,
    "dtype": "complex64"
}
#open file for writing = "w"
with open("/home/eren/Desktop/captured_signal_processed_metadata.json","w") as file123:
    #open(path,"w"), use "with" since it closes the file after opening
    json.dump(metadata,file123,indent=2)