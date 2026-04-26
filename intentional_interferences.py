import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy.signal import welch, fftconvolve, firwin, find_peaks

nperseg = 1024
fs = 2.4e6 #hz, entire bandwith
#SO SIGNALS WILL BE BETWEEN -1.2MHZ AND +1.2MHZ
N = int(4.8e6)  # 2 seconds data
t =  cp.arange(N)/fs

#AWGN Noise
awgngain = 2
awgn = awgngain * (cp.random.randn(N)+1j*cp.random.randn(N))

#WIDEBAND jammer
wb_gain = 3
numtaps = 501
centerfreq = 700e3
bw_widebandjammer = 0.8e6 
cutoff = bw_widebandjammer / 2 + 0.05*bw_widebandjammer
wb_noise = wb_gain*(cp.random.randn(N)+1j*cp.random.randn(N))
taps = firwin(numtaps, cutoff,fs=fs)
wbjammer = fftconvolve(wb_noise, taps, mode="same")
wbjammer = wbjammer*cp.exp(1j*2*cp.pi*centerfreq*t)

#CHIRP jammer: REPEATING Chirp jammer
gainchirp = 5
fstart = -0.1e6 
fstop = 0.3e6 #0.4mhz bw = 400khz 
tchirp = 0.125# 16 repeats over 2 seconds
tau = t % tchirp
k = (fstop - fstart)/tchirp 
chirp_phi = 2*cp.pi*(fstart*tau + 0.5*k*tau**2)#2pi integral(f)dt = phi, |signal| = cp.exp(1j*phi)
chirpjammer = gainchirp*cp.exp(1j*chirp_phi)*cp.exp(1j*2*cp.pi*(fstart+fstop)/2*t)

#USER USING THE BANDWITH, CARRYING DATA
user_centerfreq = -0.3e6
user_bw = 0.6e6 #from -0.6e6 to 0Hz
usergain = 4

#DEFINE USER SIGNAL AS BPSK, SIGNAL WILL BE SPECTRALLY INEFFICIENT SINCE NO PULSE SHAPING
sps = 8 #samples per symbol
Nsymbols = int(N/sps) #0.3e6
bpsk = cp.array([-1,1], dtype=cp.complex64)
index = cp.random.randint(0,2,Nsymbols) #index = 0.3e6
symbols = bpsk[index] #0.6e6
user_signal = cp.repeat(symbols,sps) #4.8e6 samples 
user_signal = usergain*user_signal*cp.exp(1j*2*cp.pi*user_centerfreq*t)

rx = user_signal + chirpjammer*0 + wbjammer*0 + awgn

freq, psd= welch(rx, fs, nperseg=nperseg,
                           window="hamming", detrend=False, return_onesided=False)

freq = cp.fft.fftshift(freq)
psd = cp.fft.fftshift(psd)

psd_dB = 10*cp.log10(psd)

noisefloor_dB = cp.median(psd_dB) # doesnt work for now since large portion of spectrum is occupied
threshold = 5 #dB
mask = psd_dB > noisefloor_dB + threshold #if more than 5dB increase in median noise floor

binwidth = fs/nperseg 
bw_occupied = cp.sum(mask)*binwidth

height = float(noisefloor_dB+threshold)
peaks_index, _ = find_peaks(psd_dB, height=height,distance=5)
#distance = 5 means, there will be one peaks per 5*binwidth = 10*2.4e6/1024 = 23.5khz

print(f"Total bandwith occupied by jammer/user data= {bw_occupied} Hz")
print(f"Peaks occured at: {freq[peaks_index]}Hz.")
print(f"Peaks were as: {psd_dB[peaks_index]} dB")

plt.figure(figsize=(10,10))
plt.title("PSD for the entire 2.4Mhz")
plt.plot(freq.get(), psd_dB.get())
plt.axhline(y=noisefloor_dB.get(),color="red",label="noise floor median")
plt.xlabel("Frequency, Hz")
plt.ylabel("PSD, dB")
plt.legend()
plt.show()
