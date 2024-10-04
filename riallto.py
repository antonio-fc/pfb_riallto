import numpy as np
import time

def pfb_fir_frontend(x, win_coeffs, M, P):
    x_p = x.reshape((M, P)).T
    h_p = win_coeffs.reshape((M, P)).T
    x_summed = np.zeros((P, M)) # x_summed = np.zeros((P, M * W - M + 1)) ???
    x_weighted = x_p * h_p
    x_summed = x_weighted.sum(axis=1)
    return x_summed.T

def fft(x_p, N):
    return np.fft.fft(x_p)
    # """
    # A recursive implementation of 
    # the 1D Cooley-Tukey FFT, the 
    # input should have a length of 
    # power of 2. 
    # """
    
    # if N == 1:
    #     return x_p
    # else:
    #     X_even = fft(x_p[::2], N/2)
    #     X_odd = fft(x_p[1::2], N/2)
    #     factor = np.exp(-2j*np.pi*np.arange(N)/ N)
        
    #     X = np.concatenate(
    #         [X_even+factor[:int(N/2)]*X_odd,
    #          X_even+factor[int(N/2):]*X_odd])
    #     return X
    
def newFFT(x, N):
    cos = np.empty(N)
    sin = np.empty(N)
    for k in range(N):
        cos[k], sin[k] = compSum(x, k, N)
    #x_pfb = cos - 1j * sin
    return (cos, sin)

def compSum(x, k, N):
    sin_sum = 0.
    cos_sum = 0.
    for n in range(N):
        cos_sum += x[n] * np.cos(k*n*2.*np.pi/N)
        sin_sum += x[n] * np.sin(k*n*2.*np.pi/N)
    return (cos_sum, sin_sum)

def squaring_pfb(x_pfb):
    return np.real(x_pfb * np.conj(x_pfb)) # same as x_psd = np.abs(x_pfb)**2

def newSquaring(x_real, x_complex):
    first = x_real * x_real
    second = x_complex * x_complex
    return first + second

def riallto_stuff(x, win_coeffs, M, P, W):
    x = x[:int(len(x)//(M*P))*M*P] # Ensure it's an integer multiple of win_coeffs
    output = np.empty([M*W, P])
    
    for i in range(0, M*W-M + 1):
        # Apply frontend, take FFT, then take power (i.e. square)
        sample_win = x[i*W:i*W+M*P]

        x_fir = pfb_fir_frontend(sample_win, win_coeffs, M, P)

        # x_pfb = fft(x_fir, P) # pfb filterbank
        real, complex = newFFT(x_fir, P)

        # x_psd = squaring_pfb(x_pfb)
        x_psd = newSquaring(real, complex)

        output[i, :] = x_psd
    

    return output

def pfb_spectrometer(x, n_taps, n_chan, n_win, n_int, win_coeffs):
    M = n_taps
    P = n_chan
    W = n_win
    
    start = time.time()
    x_psd = riallto_stuff(x, win_coeffs, M, P, W)
    time_passed = time.time() - start
    print("Time for Riallto stuff: " + str(time_passed) + "s")
    
    # Trim array so we can do time integration
    x_psd = x_psd[:np.round(x_psd.shape[0]//n_int)*n_int]
    
    # Integrate over time, by reshaping and summing over axis (efficient)
    x_psd = x_psd.reshape(x_psd.shape[0]//n_int, n_int, x_psd.shape[1])
    x_psd = x_psd.mean(axis=1)
    
    return x_psd