import numpy as np
from scipy.signal import welch,correlate
import matplotlib.pyplot as plt

#ASSUME ALL HAPPENS AFTER ANTENNA=>BPF=>LNA=>MIXER=>LPF=>ADC=> I,Q

N = int(1e6) #assuming N samples,
fs = 2e6 #assumee bw is 2e6
#N = fs*duration, so signal duration = 0.5 seconds
duration = N/fs   
t = np.arange(N)/fs
cfo = 910 #CORRELATION CAN STILL WORK FOR LOW CFO, usage of np.abs can help
preamble_length = 500

#USE CAF FOR BOTH TIME AND FREQ OFFSET
phase_off = 1j*np.pi/4
theta = np.arange(0,4,np.pi/2) #qpsk generation
qpsk = np.exp(1j*theta)*np.exp(1j*np.pi/2) #QPSK SYMBOLS GENERATION, NOT BIT ASSIGNED

preamble = np.random.choice([1+1j,-1+1j,-1-1j,+1-1j],preamble_length) #can decrease preamble length but;
#LOWER PREAMBLE LENGTH WONT CAUSE PEAKS IN CAF VS FREQ 

signalqpsk = np.zeros(len(preamble)+len(qpsk), dtype=complex) #initialize as complex
signalqpsk[0:len(preamble)] = preamble
signalqpsk[len(preamble):len(preamble)+len(qpsk)] = qpsk

awgngain = 5
awgn = awgngain*(np.random.randn(N)+1j*np.random.randn(N))

index = np.random.randint(0,N-len(signalqpsk)) #assign qpsk with time delay into awgn
awgn[index:index+len(signalqpsk)] += signalqpsk
print(f"Signal hidden into AWGN at the second: {index/fs}s")

#realistic offset
awgn = 1.03*np.real(awgn)+1j*0.98*np.imag(awgn) #IQ imbalance

#carrier freq offset, doppler
awgn = awgn*np.exp(1j*2*np.pi*cfo*t)
#phase offset
awgn = awgn*np.exp(phase_off) #awgn already carries data, awgn, cfo, phase offset

#cw jammer
A_jammer = 0.2
freqj1 = 95
freqj2 = 105
fjammer = np.arange(freqj1,freqj2,0.5)
jammer = A_jammer*np.exp(1j*2*np.pi*fjammer[:,None]*t[None,:]) #its in 2D now
jammer = np.sum(jammer, axis = 0) #make sure its (N,)

signal = awgn + jammer


#PSD of received signal
nperseg = int(2**15)#need to be able to distinguish 100hz, so freq bin less than 100hz
#deltaf=>  fs/nperseg < 100, 2e6/e2 =2e4=20.000, next 2^15,
#with 2^15 there wont be much averaging, so harsh noise spikes will be visible, but higher freq resolution. 
#can lower 2^15, since it is too high, sacrificing frequency resolution for time resolution
freq, Sxx = welch(signal, fs, window="hamming", nperseg = nperseg, detrend=False, return_onesided=False)

freq = np.fft.fftshift(freq)
Sxx = np.fft.fftshift(Sxx)
Sxx = 10*np.log10(np.abs(Sxx))
plt.figure(figsize =(10,10))
plt.subplot(3,2,1)
plt.title("Power Spectral Density")
plt.plot(freq, Sxx)
plt.xlabel("Hz")
plt.ylabel("PSD")

plt.subplot(3,2,2)
plt.title("I-Q Constellation Diagram")
plt.hist2d(np.real(signal),np.imag(signal),bins=int(np.sqrt(N))) #can use scatter for lower amount of samples

#compare normal correlation vs CAF for CFO
R_signal_preamble = correlate(signal, preamble, mode="full") #output = len(signal)+len(preamble)-1
R_signal_preamble = R_signal_preamble[len(preamble)-1:] #pull signal above noise, 
#increase SNR by 10log(len(preamble))
corr_peak_sample = np.argmax(np.abs(R_signal_preamble)) 
print(f"Time delay calculated, Cross-Correlation {float(corr_peak_sample/fs)}s") # N=fs*duration

plt.subplot(3,2,3)
plt.title("Correlation Function over preamble")
plt.plot(t, np.abs(R_signal_preamble))
plt.xlabel("s")

#SINCE THERE IS CFO, USE CAF, CROSS AMBIGUITY FUNCTION. Keeping fd low to keep complexity less
#point of CAF: look for preamble sequence within the signal
# CAF(T,F) => np.fft.ifft ( RX(f)*np.fft.fft(np.conj(preamble(f-fd))) )

#no need for fftshift, since will ifft back
signalfft = np.fft.fft(signal) #N = 1e6

fd = np.arange(-1500,1501,100)# can change fd start/stop/step size, i set this as it is to not overload RAM
#SO THAT THE MATRICES ARE THE SAME SIZE, RESIZE PREAMBLE SAME AS t!!, can use better step sizes with for loops
preamble_resized = np.zeros(N,dtype=complex) #now its same as size N
preamble_resized[0:len(preamble)] = preamble #preamble is now resized
preamble_resized = preamble_resized*np.exp(1j*2*np.pi*fd[:,None]*t[None,:]) 

preamblefft = np.fft.fft(preamble_resized,axis=1) # LEFT TO RIGHT FFT for each row
preamblefft = np.conj(preamblefft)

caf = np.fft.ifft(signalfft*preamblefft, axis=1)

caf_peak_indexes = np.argmax(np.abs(caf))

row_cfo, column_timedelay = np.unravel_index(caf_peak_indexes, np.shape(caf))

print(f"CAF predictec CFO: {fd[row_cfo]}")
print(f"CAF predicted Time delay: {column_timedelay/fs}") #since sample, N=fs*duration

plt.subplot(3,2,4)
plt.title("CAF Power vs t")
plt.plot(t, (np.abs(caf[row_cfo,:])**2)/N)

plt.subplot(3,2,5)
plt.title("CAF power vs FREQ")
plt.plot(fd, (np.abs(caf[:,column_timedelay])**2)/N)

#CFO FIXED CONST DIAGRAM:
signal = signal*np.exp(1j*2*np.pi*-fd[row_cfo]*t)
plt.subplot(3,2,6)
plt.title("Corrected Constellation Diagram based on CFO")
plt.hist2d(np.real(signal),np.imag(signal),bins=int(np.sqrt(N))) #const points will still be
#very noisy awgn since there are only 4 qpsk pts and awgn mag is 5
plt.show()