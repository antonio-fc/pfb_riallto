import npu
import numpy as np
import numpy as np
import time
import pylab as plt

from npu_stuff import runApp

def pfb_fir_frontend(x, win_coeffs, M, P):
    x_p = x.reshape((M, P)).T
    h_p = win_coeffs.reshape((M, P)).T
    x_summed = np.zeros((P, M)) # x_summed = np.zeros((P, M * W - M + 1)) ???
    x_weighted = x_p * h_p
    x_summed = x_weighted.sum(axis=1)
    return x_summed.T

def recFFT(x_r, x_c, N):
    dt = np.float32
    if N == 1:
        return (x_r, x_c)
    else:
        X_evenR, X_evenC = recFFT(x_r[::2], x_c[::2], N/2)
        X_oddR, X_oddC = recFFT(x_r[1::2], x_c[1::2], N/2)
        real_factor = np.cos(np.arange(N)*2.*np.pi/N).astype(dt)
        complex_factor = np.sin(np.arange(N)*2.*np.pi/N).astype(dt)
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

def newSquaring(x_real, x_complex):
    first = x_real * x_real
    second = x_complex * x_complex
    return first + second

def riallto_stuff(x, win_coeffs, M, P, W, app):
    x = x[:int(len(x)//(M*P))*M*P] # Ensure it's an integer multiple of win_coeffs
    output = np.empty([M*W, P])
    dt = np.float32
    
    for i in range(0, M*W-M):
        # Apply frontend, take FFT, then take power (i.e. square)
        sample_win = x[i*W:i*W+M*P]

        gg = runApp(app, sample_win, win_coeffs, np.zeros(shape=(M*P), dtype=dt), i)
        
        x_fir = pfb_fir_frontend(gg, win_coeffs, M, P)

        real_pfb, complex_pfb = recFFT(x_fir, np.zeros(P, dtype=dt), P)

        x_psd = newSquaring(real_pfb, complex_pfb)

        output[i, :] = x_psd
    

    return output

def pfb_spectrometer(x, n_taps, n_chan, n_win, n_int, win_coeffs, app):
    M = n_taps
    P = n_chan
    W = n_win
    
    start = time.time()
    x_psd = riallto_stuff(x, win_coeffs, M, P, W, app)
    time_passed = time.time() - start
    print("Time for Riallto stuff: " + str(time_passed) + "s")
    
    # Trim array so we can do time integration
    x_psd = x_psd[:np.round(x_psd.shape[0]//n_int)*n_int]
    
    # Integrate over time, by reshaping and summing over axis (efficient)
    x_psd = x_psd.reshape(x_psd.shape[0]//n_int, n_int, x_psd.shape[1])
    x_psd = x_psd.mean(axis=1)

    print("DONE!")
    return x_psd