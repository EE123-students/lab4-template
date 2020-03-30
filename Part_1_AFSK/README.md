# Lab 4 Part I: Digital Communication with Audio Frequency Shift Keying (AFSK)

In this part of the lab we are going to experiment with digital modulation and communication. Network Communication systems have layered architecture. The bottom layer is the physical layer that implements the modulation. In this lab we will experiment Audio-Frequency-Shift-Keying [AFSK](http://en.wikipedia.org/wiki/Frequency-shift_keying), which is a form of Binary Frequency Shift Keying (BFSK) in the audio range (hence the 'A'). In the lab we will implement a modulator/demodulator for AFSK.  

In Part I of the lab we will leverage [AX.25](http://www.tapr.org/pub_ax25.html), which is an amateur-radio data-link layer protocol. [AX.25](http://www.tapr.org/pub_ax25.html) is a packet based protocol that will help us transmit data using packets. It implements basic synchronization, addressing, data encapsulation and some error detection.  In the ham world, an implementation of AFSK and [AX.25](http://www.tapr.org/pub_ax25.html)  together is also called a [TNC ( Terminal Node Controller )](http://en.wikipedia.org/wiki/Terminal_node_controller). In the past, TNCs were separate boxes that hams would attach to their radios to communicate with packet-based-communication. Today, it is easy to implement TNCs in software using the computer's soundcard.... as you will see here! 





```python
# Import functions and libraries
import numpy as np
from numpy import r_
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, title,stem, loglog, xlabel, ylabel
import bitarray, time, urllib, ssl
from scipy import signal, integrate
from math import gcd
from functools import reduce
from numpy.random import rand, randn
from numpy import ones,zeros, pi, cos, exp, sign
from lab4functions import *
%matplotlib inline
```

## AFSK1200, or Bell 202 modem

AFSK1200 encodes digital binary data at a data-rate of 1200 b/s. It uses the frequencies 1200 Hz and 2200 Hz ( center frequency of $1700$ Hz $\pm 500$ Hz) to encode the '0's and '1's (also known as space and mark) bits.  Even though it has a relatively low bit-rate it is still the dominant standard for amateur packet radio over VHF. It is a common physical layer for the AX.25 packet protocol and hence a physical layer for the  Automatic Packet Reporting System (APRS), which we will describe later. 

The exact analytic frequency spectrum of a general FSK signal is difficult to obtain. But, when the mark and space frequency difference $\Delta f$ is much larger than the bit-rate, $B$,  then the bandwidth of FSK is approximately $2\Delta f + B$. This is not exactly the case for AFSK1200 where the spacing between the frequencies is 1000 Hz and the bit-rate is 1200 baud.

<center><img src="AFSK.png" alt="AFSK" style="width: 350px;"/></center>
<center>Figure 1: Approximate spectrum of AFSK</center>

Note, that for the (poor) choice of 1200/2200 Hz for frequencies, a synchronous phase (starting each bit with the same phase) is not going to be continuous because at 1200 bit rate the 2200 Hz tone is not completing full two cycles (a choice of 2400 would have been ideal!... but this is the standard we are forced to use for compatability with other systems). For the Bandwidth to be narrow, it is important that the phase in the modulated signal is continuous. Any discontinuity in the phase will result in high frequency content. For this reason, AFSK1200 has to be generated in the following way:
$$ s(t) = \cos\left(2\pi f_c t + 2\pi \Delta f \int_{-\infty}^t m(\tau)d\tau \right),$$
where $m(t)$ has the value 1 for the duration of a mark bit, and a value -1 for the duration of a space bit. Such a signal $m(t)$ is called an Non-Return-to-Zero (NRZ) signal in the digital communication jargon.  Here's a link to some relevant [notes](http://www.dip.ee.uct.ac.za/~nicolls/lectures/eee482f/13_fsk_2up.pdf) provided by Fred Nicolls from the University of Cape Town.

The integration guarantees that the phase is continuous. In addition, the instantaneous frequency of $s(t)$ is the derivative of its phase, $2\pi f_c + 2\pi \Delta f m(t)$, which is exactly what we need. 

#### Task 1:

* Write a function `sig = afsk1200(bits,fs)` the function will take a bitarray (bitarray is a python module for storing array of bits) of bits, and a sampling rate.  It will output an AFSK1200 modulated signal of them, sampled at `fs` [Hz]. **Note that Mark frequency is 1200Hz and Space Frequency is 2200 Hz**. 

* Note that when `fs` does not divide by 1200, each "bit" will have non-integer length in samples. If you are not careful, this would lead to deviation from the right rate over time. To make sure that you produce signals that have the right rate over time generate the signal first at a rate of `lcm((1200,fs))`, the least common multiple (which does divide by 1200) for the entire bit sequence and then downsample by the right value at the end. You don't necessarily need to low-pass filter, since the signal is narrow band anyways. 

* For integration, use the function `integrate.cumtrapz`, which implements the trapezoid method. 


```python
def afsk1200(bits, fs = 48000):
    # the function will take a bitarray of bits and will output an AFSK1200 modulated signal of them, sampled at fs Hz
    #  Inputs:
    #         bits  - bitarray of bits
    #         fs    - sampling rate
    # Outputs:
    #         sig    -  returns afsk1200 modulated signal samples at fs Hz
    
    # your code here
    
    
    
    return sig
        
              
```

To help you debug, we generated a sequence of 4096 bits and its AFSK1200 encoded signals at sampling-rates of 8192Hz, 11025Hz, 44100Hz and 48000Hz. Compare the result of your function and make sure they match well. We assume that the phase at n=0 is zero (i.e., cosine modulation). Use the following code to load the data: 


```python
# ------------- code we used to generate the bitstream and the signals ------------------
#bits_ref=bitarray.bitarray((rand(4096)>0.5).tolist())
#sig_ref8192 = afsk1200(bits_ref,fs = 8192)
#sig_ref11025 = afsk1200(bits_ref,fs = 11025)
#sig_ref44100 = afsk1200(bits_ref,fs = 44100)
#sig_ref48000 = afsk1200(bits_ref,fs = 48000)
#np.savez_compressed('debug_ref.npz', bits=bits_ref, sig_ref8192=sig_ref8192, sig_ref11025 = sig_ref11025,sig_ref44100=sig_ref44100, sig_ref48000=sig_ref48000)


# Load the file from the class website
#testfile = urllib.request.URLopener()
#testfile.context = ssl._create_unverified_context()
#%testfile.retrieve("https://inst.eecs.berkeley.edu/~ee123/sp16/lab/lab5/debug_ref.npz", 'debug_ref.npz')

testnpz = np.load('./debug_ref.npz')
bytes_ref = bytes(testnpz['bits'])
bits_ref = bitarray.bitarray()
bits_ref.frombytes(bytes_ref)
sig_ref8192 = testnpz['sig_ref8192']
sig_ref11025 = testnpz['sig_ref11025']
sig_ref44100 = testnpz['sig_ref44100']
sig_ref48000 = testnpz['sig_ref48000']

# Check that all the loaded signals align
fig = figure(figsize=(16,4))
plt.plot(r_[0.0:148]/8192,sig_ref8192[:148])
plt.plot(r_[0.0:200]/11025,sig_ref11025[:200])
plt.plot(r_[0.0:800]/44100,sig_ref44100[:800])
plt.plot(r_[0.0:870]/48000,sig_ref48000[:870])
plt.title('AFSK1200 with different sampling rates')
plt.legend(('8192Hz','11024Hz','44100Hz', '48000Hz'))

```

#### Task 2:

Apply your function on the bitstream above with the different sampling rate. Validate that they match well:



```python
# Validation code
# bits_ref=bitarray.bitarray((rand(4096)>0.5).tolist())
bytes_ref = bytes(testnpz['bits'])
bits_ref = bitarray.bitarray()
bits_ref.frombytes(bytes_ref)

sig_ref8192 = afsk1200(bits_ref,fs = 8192)
sig_ref11025 = afsk1200(bits_ref,fs = 11025)
sig_ref44100 = afsk1200(bits_ref,fs = 44100)
sig_ref48000 = afsk1200(bits_ref,fs = 48000)

# Your validation code here
# Check that all the loaded signals align




```

### Spectrum of AFSK1200

We will now generate an AFSK modulated signal and compute its spectrum

#### Task 3:
* Generate a new sequence of 4096 random bits with equal probability, using the code     `bitarray.bitarray((rand(4096)>0.5).tolist())` 
* Generate the AFSK1200 signal at 12KHz 
* Compute the average power-spectrum with a spectral resolution of 10Hz (What's the window size?)
* Display the result between 0 and 3KHz, using the command `plt.axis`
* Does the spectrum look like the one in Figure 2?




```python
fs = 12000
bits=bitarray.bitarray((rand(4096)>0.5).tolist())


# your code here



```

## AFSK1200 demodulation
AFSK is a form of digital frequency modulation. As such, it can be demodulated like FM. However, this is not the best way to demodulate it in practice. For example, any tone interference between the mark and space frequency will break the demodulation.  

Because AFSK alternates between two frequencies, we can place two bandpass filters around the frequency of the Mark and Space and use envelope detection to determine which frequency is active in a bit period. This is called a non-coherent AFSK demodulation, because the receiver phase does not need to be synced to the transmitter phase in order to demodulate the signal. The implementation we will use here is loosely based on on the one by Sivan Toledo (4X6IZ), a CS faculty at Tel-Aviv University who has written a nice article on a high-performance AX.25 modem, and also loosly based on DireWolf, a multi-platform software TNC. You can find Sivan's article [Here](http://www.cs.tau.ac.il/~stoledo/Bib/Pubs/QEX-JulAug-2012.pdf), and Direwolf GitHub link [Here](https://github.com/wb2osz/direwolf). 


### Non-Coherent Demodulation of AFSK
Here's a diagram of a non-coherent AFSK1200 demodulator that returns an NRZ signal:
<br><br><br>
<center><img src="AFSK_demod.png" alt="AFSK" style="width: 950px;"/></center>
<br>
<center>Figure 3: AFSK1200 non-coherent demodulator</center>


#### Task 4:

In this part of the lab we will be using a sampling rate of 12 KHz as as it divides well with 1200 (12000/1200 = 10 samples per bit). It is also ready to be easily played on the soundcard. 
As mentioned in the article, it is recommended to bandpass filter before processing  (900-2500Hz). This filter has no theoretical benefit for the case of random gaussian noise. But it still has some advantages when the noise and interferences are non-gaussian. We will not do it now, but will implement this later when we implement our modem.

* Using signal.firwin, design a TBW=2 LP filter with a **(two-sided)** bandwidth of approximately 1200 Hz (highest frequency 600 Hz). It's easier if you choose the number of taps to be odd. (what's the length of the filter?)
* From the LP filter generate two bandpass filters by complex modulating the LP filter to be centered around 1200 Hz and 2200 Hz respectively. 
* Filter the random stream of bits you generated previously using the two filters. 

The absolute value of the result represents the envelope of the filtered signal. The difference between the envelopes should represent the NRZ signal. 

* Plot the result of the envelope detection of the mark and space signals for the first 1200 samples on the same plot. Can you see how they switch?
* Compute the "analog" NRZ signal by taking the difference between the mark and space envelopes. I refer to this as "analog" since it is not purely binary.
* To implement the optional filter (Fig. 1), filter the NRZ signal, now, with a **low-pass filter**. Have the filter be the same length you used for the mark and spaces, only with a cutoff frequecy of 1200\*1.2 Hz. This is a balance between noise rejection and keeping the signal and helps a bit with detection in noise. 
* In a different plot, display the filtered NRZ for the first samples corresponding to 0.1 seconds. Can you see the bits?


#### \*\*\* How are parameters fine tuned? 
Well... the answer is complicated. There's theory and practice. From a theory point of view, we choose the right bandwidths and signal lengths. From a practical point of view, parameters can be tuned by testing performance over simulations and real-life experiments. For example, WA8IMF created a test CD for testing AFSK1200 modems. He recorded many minutes of APRS traffic in LA. Hardware TNC can usually decode 500-600 packets on these test CD's whereas optimized software TNCs can detect over 1000. Here's a [link](http://wa8lmf.net/TNCtest/) for the CD.

After we implement our APRS receiver, you could play with parameters to improve your packet decoding performance. 




```python
fs = 12000
TBW = 2
BW = 1200

# your code here


```

### Eye diagram: A timing, and noise margin visualization tool

When looking at the NRZ output, you can see it oscilates between positive value, which represent the a "mark" bits and negative values which represent the "space" bits. Instead of plotting it as a long signal, we can plot it in an osciloscope where the signal is applied to the vertical axis, and the horizontal axis period is bit-length -- triggered at bit transition. This means that zero-crossing will display at the edges of the plot, while the point in which we sample will display in the middle. Below is an example showing the concept: 


<center><img src="Eye.jpeg" alt="Eye diagram" style="width: 1200px;"/></center>
<center>Figure 4: Eye diagram is used to visualize timing mismatches, and noise margin</center>

Since the sampling rate is 12000 Hz, and the bit-rate is 1200b/s, we get 10samples / bit. This is not a perfect choice, because the best sampling should be done at the 4.5 Index-- but in practice it is not a huge issue. We can therefore sample at the 4th or 5th sample, or the average of both. Alternatively, using a filter with an even number of taps will result in an additional half sample delay which will fix the issue. The figure also shows the result when the sampling period is delayed w.r.t to the signal. In that case, we sample at an offset and lose noise margin. We can see the effect of noise on the eye diagram. Since decisions are made based on the sign of the sample, we can see how robust digital commnication can be. Even in this case where noise exists, we will be able to correctly decode the bit value. 

#### Task 5: 

Create an eye diagram from the analog NRZ signal you got, overlaying the signals for the first 200 bits (2000 samples). What is the index you should start sampling from?


```python
# your code for displaying an eye diagram:


```

#### Task 6:
* Extract the digital NRZ signal by computing the signum (`sign`) function of the "analog" NRZ. 
* The bit value is the value of the NRZ function in the middle of the bit period.  **Don't forget to compensate for the delay of the filters**, which is half the filter length, or use `mode='same'`. Decode the bits and store them as a `bitarray` type.
* Plot 800 samples of the Digital NRZ. Overlay a stem plot on top of that at the indexes in which you sampled the bit values. Make sure as a sanity check that you actually sampled at the middle of the interval. Only show the plot for the first 20 bits. At 12000 KHz sampling and bit rate of 1200 Hz, each bit is 10 samples long
* Print the value of the first 100 decoded bits and compared to the encoded ones. Make sure they are exactly the same!



```python
%matplotlib inline



# your code here:




# print the decoded bits compared to the transmitted bits








bit_dec = bitarray.bitarray((NRZ[idx]>0).tolist())
print(bit_dec[:100])
print(bits[:100])
```

#### Task 7:

* Write a function NRZ = nc_afsk1200Demod(sig, fs=fs, TBW=TBW) that implements the above non-coherent demodulation and returns the "analog" NRZ (i.e. without rectifying it). 


```python
def nc_afsk1200Demod(sig, fs=48000.0, TBW=2.0):
    #  non-coherent demodulation of afsk1200
    # function returns the NRZ (without rectifying it)
    # 
    # sig  - signal
    # baud - The bitrate. Default 1200
    # fs   - sampling rate in Hz
    # TBW  - TBW product of the filters
    #
    # Returns:
    #     NRZ 
    
   # your code here
    
    
    
    
    
    
    
    
    
    
    
    
    return NRZ

```

### Bit Error Rate (BER)
One way to evaluate the properties of a digital modulation scheme is to compute the bit-error-rate (BER) curves as a function of signal-to-noise ratio (SNR). The BER is the number of bit errors (received bits that have been altered due to decoding error) divided by the total number of transmitted bits. 

Let's calculate the BER for our AFSK demodulator:

#### Task 8:
* Generate a 10000 long random bitstream
* AFSK1200 modulate the bitstream
* Add random gaussian noise with a standard deviation of 0.75 to the AFSK signal. 
* Demodulate  
* Plot the first 640 samples of the output analog NRZ of the demodulation (64 bits at 12 KHz), and overlay a stem plot with stems at the center of bits period -- look at the result. 
* Make an eye diagram of the same number of bits. Can you see why digital communication is robust?
* Compute the BER by comparing the bitstream before and after modulation/demodulation



```python
#Your code here:






# plot the first 640 samples of the analog NRZ and overlay a stem plot with stems at the center of bits period




# make an eye diagram of the bits from 640 samples





# print the decoded bits compared to the transmitted bits


bit_dec = bitarray.bitarray((NRZ[idx]>0).tolist())
print(bit_dec[:64])
print(bits[:64])
```


```python
# Calculate BER:






```

Your bit error rate should be around 0.03. It will depend also on the quality of your demodulator. You can try to repeat the experiment for different choices of filters if you like. 

### Computing BER curves

BER curves are usually displayed in log log of the BER vs SNR. SNR is measured by energy per bit over noise power spectral density.
Since we are just interested in the trend, we will plot the BER vs $1/\sigma$, where $\sigma$ is the noise standard deviation. 

To help you debug your code, we generated our own curves and saved the results. Your results should be similar to ours. 

#### Task 9:

* Repeat the experiment for the range $\sigma=$ np.array([0.2,0.25,0.3,0.35,0.4,0.5,0.65,0.8,1.0,1.5,2,2.6,3.5,5])
* Use the function loglog to plot the BER as a function of 1/$\sigma$. What's the BER at really low-SNR? What does it mean about the information the channel is carrying in low-SNR?
* **This code took 75 seconds on my raspberry pi to run. Go get a drink....** 


```python
### Load our simulation result:
testnpz = np.load('BER_ref.npz')
BER_ref = testnpz['BER']
sigma_vec = np.array([0.2,0.25,0.3,0.35,0.4,0.5,0.65,0.8,1.0,1.5,2,2.6,3.5,5])





# Your code:











# plot


loglog(1/sigma_vec,BER_nc[:len(sigma_vec)])
loglog(1/sigma_vec,BER_ref[:len(sigma_vec)],'r')

plt.legend(('mine','Miki''s'))

title("empirical BER for AFSK demodulation")
xlabel("SNR")
ylabel("BER")






```

### Timing Recovery

One of the most important part of digital demodulation is the synchronization between the transmitter and receiver. We would like to sample the NRZ signal at the peak, which happens in the middle of the bit interval. However, we don't necessarily know when the transmitter starts sending the bits. There could also be some offset with the bit-rate between the transmitter and receiver due to the different clocks in the systems

There are many ways to do this. For example, if there's a known preamble sequence we could look for, we can perform matched filtering with the known sequence to find the bit rate and synchronize to its start. 

Here we will use a simple, yet elegant solution, that was implemented in DireWolf, which uses a counter based phased-lock-loop (PLL). Here's a system diagram for the timing recovery (D is a sample delay):

<center><img src="PLL.png" alt="PLL" style="width: 550px;"/></center>
<center>Figure 4: Timing Recovery</center>

The idea is simple. For each incoming sample we advance a 32bit signed counter by $2^{32}/(f_s/\text{baud})$. The counter will overflow exactly every $f_s/\text{baud}$ samples, which is the number of bits per symbol. When overflow happens, we record the sign of the NRZ signal and output it. If the counter is synchronized to the NRZ signal, whenever there's a zero-crossing of the NRZ signal due to bit-sign change there also should be a zero-crossing of the counter and the counter should overflow in the middle of a symbol. 

So, in order to synchronize the PLL counter to the NRZ signal, whenever the NRZ signal has a zero-crossing, we will "nudge" the counter by multiplying it with $0<a<1$ to get it closer to zero. When the PLL is locked, the zero-crossings would match and the "nudging" would do nothing. When the PLL is not locked, then "nudging" would move the counter towards the right offset. This solution relies on the data not to have long sequences of the same bit value-- because long sequences of the same bit will not have zero-crossings in the NRZ. As we will see later, APRS (and its underliying protocol AX.25) solved this by injecting bit transitions in a process called "bit stuffing". Also, before each packet, it is common to send a string of alternating values for helping receiver synchronization. 

The value of $a$ is of course a tradeoff. When $a$ is small, the PLL will lock fast, but would be sensitive to jitter due to noise. In that case, whenever the NRZ signal would have a zero-crossing the PLL would nudge a lot, and therefore be sensitive to noise as well.  When $a$ is closer to 1, the PLL will have less jitter, but would take more time to lock.  A value of $a=0.75$ seems to work well, to balance these two. 

#### Task 10:

Write a function `idx = PLL(NRZa, a, fs = fs, baud=baud)`. 

The function should take as input a segment of an "analog" NRZ sequence, the "nudge" factor $a$, the sampling rate and the baud rate. It should return the indexes corresponding to the middle sampling points of symbols.

* Python types don't allow for overflow, so when incrementing the counter, cast it to `int32`, to force overflow. 


```python
def PLL(NRZa, a = 0.75 , fs = 48000, baud = 1200):
  
# 
    # function implements a simple phased lock loop for tyming recovery
    #
    # Inputs:
    #          NRZa -   The NRZ signal
    #          a - nudge factor
    #          fs - sampling rate (arbitrary)
    #          baude  - the bit rate
    #
    # Outputs:
    #          idx - array of indexes to sample at
    #
    
    
    # Your code here:
    
    
    
 
    
    return idx[]
    
```

The following code generates 4 AFSK1200 packets of length 24, sampled at 48000 Hz. Each packet consistes of  1,0,1,0,1,0 training sequence followed by 18 bits. It also puts random spacings between packets as well as noise.

To help you debug, we generated this packet, modulated, demodulated and computed the timing using our own implementation of the PLL.  You can use our data to see if your timings are correct. 

#### Task 11:
* Run your PLL and this data with $a = 0.75$
* Plot the NRZa signal and overlay the sampling points that were returned by the PLL. Make sure the PLL works! You should see that the PLL will lock within very few bit lengths
* Create an eye diagram for the last 18 bits of the last packet, plotting the signal starting 20 samples before the index, to 20 samples after with a new trigger for each index. 

**Note that for our data, on the 2nd and 4th packets, the PLL takes 5 bit lengths to lock**



```python
#bits=bitarray.bitarray([True,False,True,False,True, False]+(np.random.randn(24)>0).tolist())
#sig = zeros(np.int32(np.random.rand(1)*200+200))
#sig = np.concatenate((sig,afsk1200(bits, fs=48000)))

#for n in range(0,3):
#    bits=bitarray.bitarray([True,False,True,False, True, False]+(np.random.randn(18)>0).tolist())
#    sig = np.concatenate((sig,zeros(np.int32(np.random.rand(1)*200+200)),afsk1200(bits, fs=48000)))
#sig = sig + 0.1*np.random.randn(len(sig))
#NRZ = nc_afsk1200Demod(sig, fs=48000)
#idx  = PLL(NRZ, a=0.75)
#np.savez_compressed('debug_pll.npz', idx=idx, sig = sig, bits = bits, NRZ=NRZ)


```


```python

testnpz = np.load('./debug_pll.npz')
bits= testnpz['bits']
sig = testnpz['sig']
idx = testnpz['idx']
NRZ = testnpz['NRZ']

figure(figsize=(16,4))
plot(NRZ)
stem(idx,NRZ[idx.astype(int)])



Nbits = 40
figure(figsize=(16,4))
for n in range(len(idx)-1-18,len(idx)-1):
    plot(r_[-Nbits//2:Nbits//2],NRZ[int(idx[n]-Nbits//2):int(idx[n])+Nbits//2])

# Run your PLL on the NRZ and compare the idx you get with ours. They should be similar. 




```


```python
# your code for running your PLL Below here:



```

### Timing Jitter

Let's see the effect of the PLL nudging factor `a` on the timing jitter. 

#### Task 12:
* Generate a sequence of 1000 random bits 
* Modulate the signal using AFSK1200 at 48000 sampling rate
* Add 4000 zero samples in the begining so the PLL will need to lock.
* Add random gaussian noise with standard deviation of 0.75

First the PLL will be confused by the noise -- and should be all over the place for about 100 indexes (4000 samples / 40 samples/bit). Once it is locked, it should output indexes with increments around 40. 
Even though there's noise, the best sampling timing is every 40 samples ( $f_s = 48000$ ), because that's the original rate without noise. If our PLL is insensitive to noise, then it should give us this value all the time.

* Used the PLL with $a=0.95$, to compute the indices of sampling points. Then, compute the finite difference in the indexes. Ideally, it should be 40 all the time.
* Repeat the above for $a=0.75$, and $a=0.4$



```python
# You get Task 12 for free:

bits=bitarray.bitarray((np.random.randn(1000)>0).tolist())
sig = afsk1200(bits, fs=48000)
sig=np.concatenate((zeros(4000),sig))
sig = sig + np.random.randn(len(sig))*0.75

NRZ = nc_afsk1200Demod(sig, fs=48000)
idx1 = PLL(NRZ, a=0.9)
idx2 = PLL(NRZ, a=0.75)
idx3 = PLL(NRZ, a=0.4)



fig = plt.figure(figsize=(16,2))
plot(np.diff(idx1))
title('Jitter for a=0.95')
plt.axis((0,1000,20,50))
fig2 = plt.figure(figsize=(16,2))
plot(np.diff(idx2))
plt.axis((0,1000,20,50))
title('Jitter for a=0.75')
fig3 = plt.figure(figsize=(16,2))
plot(np.diff(idx3))
plt.axis((0,1000,20,50))
title('Jitter for a=0.4')
```

Now, you are ready to move to Part II, the hardware component of this lab.


```python

```
