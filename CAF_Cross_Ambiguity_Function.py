import numpy as np
import matplotlib.pyplot as plt

fs = 1e3
N = 1024#amount of samples
t = np.arange(N)/fs #1024,
#signal duration is N/fs seconds
#each sample carries 1/fs seconds of data
#so 1/fs * 17 = signal is present for 0.017s in an entire 1 sec
#INCREASE PREAMBLE TO DISTINGUISH CFO EASIER or can use chirps like radars & LoRa
r_t = np.fromfile("/home/eren/Desktop/captured_signal.dat",dtype=np.complex64) #wont overload RAM(smalldata)
r_t = r_t[:N] #after np.complex64, np knows its interleaved 32bit float I and Q 
s_t = [1,1,-1,-1,1,1,-1,1,1,1,-1,-1,1,-1,1,1,1] # len(s_t) = 17, BARKER CODE BETTER 
#r_t = 0.5*(np.random.randn(N)+1j*np.random.randn(N)) #shape (1024,) 
#un-hashtag the line above if dont want to use fromfile 
fd = np.arange(0,30,1) #define fd to guess CFO, #shape (30,)

index = np.random.randint(0, N-len(s_t)) #since if N or s(t) changes, 
cfo = np.random.randint(0,30)

r_t[index:index+17] += s_t #reference signal buried under noise for 17 samples, 17/1024
r_t = r_t*np.exp(1j*2*np.pi*cfo*t)
#AWGN + SİGNAL BURİED WİTH TİME DELAY(t[index]) and CFO(cfo), SO REALISTIC SIGNAL

timedelay = t[index]
print(f"This is the time delay randomly set: {timedelay}s")
print(f"This is the CFO randomly set: {cfo}Hz")

def calculate_caf(r_t, s_t, fd, t):

    r_tfft = np.fft.fft(r_t) #1024, 1D FFT
    
    N = len(r_t) #(1024,)
    s_padded = np.zeros(N, dtype=complex)  #(1024,)
    s_padded[:len(s_t)] = s_t 
    s_tmatrix = s_padded*np.exp(1j*2*np.pi*fd[:,None]*t[None,:]) # (30,1) * (1,1024)= (30,1024) matrix
    
    s_tfft = np.fft.fft(s_tmatrix, axis=1) #axis 1 IS LEFT TO RIGHT, (30,1024)
    s_tfft = np.conj(s_tfft) #conjugate of reference signal 

    caf = np.fft.ifft(r_tfft*s_tfft,axis=1) #look for time shifted version of s inside rx signal
    #s_tfft is 30, 1024 . r_tfft is (1024,) 
    #After, numpy treats (1024,) as 1,1024 and copies that row downwards 30 times 

    caf_power = (np.abs(caf)**2)/N
    cfo_est_index, delay_est_index = np.unravel_index(np.argmax(caf_power), np.shape(caf_power))
    cfo_est = fd[cfo_est_index]
    delay_est = t[delay_est_index]

    papr = np.max(caf_power[cfo_est_index,delay_est_index])/np.mean(caf_power)

    plt.figure(figsize=(20,20))
    plt.subplot(2,1,1)
    plt.plot(fd, caf_power[:,delay_est_index], color="blue",label="Caf power vs f")
    #due to gabor limit, might seem wide since 17 sample pulse duration is short 
    #because of the sample rate, fs(1e3)  
    plt.xlabel("Freq (Hz)")
    plt.ylabel("Caf Power")
    plt.legend()
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(t, caf_power[cfo_est_index,:], color="red",label="Caf power vs t")
    plt.xlabel("Time (s)")
    plt.ylabel("Caf Power")
    plt.grid(True)
    plt.legend()
    plt.show()

    caf_power = caf_power[cfo_est_index,delay_est_index]
    return cfo_est, delay_est, caf_power, papr 


cfo_est, delay_est, caf_power, papr = calculate_caf(r_t, s_t, fd, t)

print(f"CFO CALCULATED:{cfo_est}.")
print(f"TIME DELAY CALCULATED:{delay_est}.")
print(f"Cross Ambiguity Function, CAF Power:{caf_power}.")
print(f"Peak to Avg Power Ratio, PAPR:{papr}.")

