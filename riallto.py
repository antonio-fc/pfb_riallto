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

def sinRecFFT(x, N):
    if N == 1:
        return x
    else:
        X_even = sinRecFFT(x[::2], N/2)
        X_odd = sinRecFFT(x[1::2], N/2)
        sin_factor = np.sin(np.arange(N)*2.*np.pi/N)
        # factor = np.exp(-2j*np.pi*np.arange(N)/ N)
        
        X = np.concatenate(
            [X_even+sin_factor[:int(N/2)]*X_odd,
             X_even+sin_factor[int(N/2):]*X_odd])
        return X
    
def cosRecFFT(x, N):
    if N == 1:
        return x
    else:
        X_even = cosRecFFT(x[::2], N/2)
        X_odd = cosRecFFT(x[1::2], N/2)
        cos_factor = np.cos(np.arange(N)*2.*np.pi/N)
        # factor = np.exp(-2j*np.pi*np.arange(N)/ N)
        
        X = np.concatenate(
            [X_even+cos_factor[:int(N/2)]*X_odd,
             X_even+cos_factor[int(N/2):]*X_odd])
        return X
    
def recFFT(x_r, x_c, N):
    if N == 1:
        return (x_r, x_c)
    else:
        X_evenR, X_evenC = recFFT(x_r[::2], x_c[::2], N/2)
        X_oddR, X_oddC = recFFT(x_r[1::2], x_c[1::2], N/2)
        real_factor = np.cos(np.arange(N)*2.*np.pi/N)
        complex_factor = np.sin(np.arange(N)*2.*np.pi/N)
        # factor = real_factor - 1j*complex_factor
        # factor = np.exp(-2j*np.pi*np.arange(N)/ N)
        
        a, b, c, d = X_oddR, X_oddC, real_factor, complex_factor
        real1 = a*c[:int(N/2)] - b*d[:int(N/2)]
        real2 = a*c[int(N/2):] - b*d[int(N/2):]
        complex1 = b*c[:int(N/2)] + a*d[:int(N/2)] 
        complex2 = b*c[int(N/2):] + a*d[int(N/2):]

        X_r = np.concatenate(
            [X_evenR+real1,
             X_evenR+real2])
        X_c = np.concatenate(
            [X_evenC+complex1,
             X_evenC+complex2])
        return (X_r, X_c)
    
def newFFT(x, N):
    cos = np.empty(N)
    sin = np.empty(N)
    for k in range(N):
        cos[k], sin[k] = compSum2(x, k, N)
    #x_pfb = cos - 1j * sin
    return (cos, sin)

def compSum(x, k, N):
    sin_sum = 0.
    cos_sum = 0.
    for n in range(N):
        cos_sum += x[n] * np.cos(k*n*2.*np.pi/N)
        sin_sum += x[n] * np.sin(k*n*2.*np.pi/N)
    return (cos_sum, sin_sum)

def compSum2(x, k, N):
    sin_sum = np.empty(N)
    cos_sum = np.empty(N)
    for n in range(N):
        cos_sum[n] = x[n] * np.cos(k*n*2.*np.pi/N)
        sin_sum[n] = x[n] * np.sin(k*n*2.*np.pi/N)
    return (np.sum(cos_sum), np.sum(sin_sum))

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
        real, complex = recFFT(x_fir, np.zeros(P), P)
        # real = cosRecFFT(x_fir, P)
        # complex = sinRecFFT(x_fir, P)
        # real, complex = newFFT(x_fir, P)
        # x_pfb = real - 1j*complex

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