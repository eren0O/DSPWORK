#CALCULATE EB/N0 OF A SIGNAL
# Rb = log2(M)*Rs*code_rate 
import cupy as cp
import matplotlib.pyplot as plt

#ASSUME RECEIVED A SIGNAL OF 1 SECOND
duration = 1
fs = 2.4e6 #bandwith of SDR
N = int(fs*duration) #total samples in the BW
M = 8  #since M-PSK and 8-PSK
#BIT ASSIGNMENT ON THE 8-PSK 
sps = 8 #samples/symbol
theta = cp.arange(8)*cp.pi/4 #set to 8 for now to generate 8PSK for one constellation 
psk8 = cp.exp(1j*theta)
#total amount of symbols caught in the bw of 8PSK = int(N/sps) = 2.4e6 /8 = 0.3e6
Nsymbols =  int(N/sps) #samples / (samples/symbol) = symbols
index = cp.random.randint(0,M,Nsymbols) #each index stands for what symbol will the
#index be representing, 3 bits per symbol since 2^3=8-PSK

symbols = psk8[index] #assign the symbol indexes to the constellation diagrams symbols
#if index[0]=5, symbol[0] = psk8[5], which corresponds to the sixth symbol. 
# This is basicaly CCW bit assignment, but IRL: Gray coding. Each 3-bit is assigned to the symbol.
# BITS TO SYMBOLS BASED ON THE MODEM DICTIONARY LOOKUP 

tx = cp.repeat(symbols, sps) #signal[0]= symbol[0] for 8 cycles, then signal[8]=symbol[1]...
#so receiver sees that symbols only change every /8 times the transmitters processing rate
#USING PULSE SHAPING FILTER HERE IS CRITICAL to not ruin freq domain, 
#if not used sps, signal wont be the same as fs value 

awgngain = 0.5
awgn = awgngain*(cp.random.randn(N)+1j*cp.random.randn(N))

signal = awgn + tx

#steps: EVM=> SINR=1/EVM**2 => Eb/N0 = SINR*B/Rb

#Pavgsignal is 1 already since 8-PSK  
evm = cp.sqrt(cp.mean( 
    (cp.real(signal[0::sps])-cp.real(symbols))**2 #CATCH SIGNAL[ ] EVERY 8 SYMBOLS
    + (cp.imag(signal[0::sps])-cp.imag(symbols))**2 
))

sinr = evm**-2
B = fs #bandwith for real world applications isn't fs, its the occupied BW or the noise BW
Rs = fs/sps
Rfec = 8/8
Rb = Rs*cp.log2(M)*Rfec 
EbNO = sinr*B/Rb
EbNO = 10*cp.log10(EbNO)
print(f"Energy per bit divided by Noise PSD Eb/N0: {EbNO}dB")

plt.hist2d(cp.real(signal).get(),cp.imag(signal).get(),bins=200)
plt.show()
