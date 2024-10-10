import npu
import numpy as np
import numpy as np
import time
import pylab as plt

from npu_stuff import runApp

def fft(x_p, N):
    return np.fft.fft(x_p)

def squaring_pfb(x_pfb):
    return np.real(x_pfb * np.conj(x_pfb)) # same as x_psd = np.abs(x_pfb)**2

def pfb_fir_frontend(x, win_coeffs, M, P):
    x_p = x.reshape((M, P)).T
    h_p = win_coeffs.reshape((M, P)).T
    x_summed = np.zeros((P, M)) # x_summed = np.zeros((P, M * W - M + 1)) ???
    x_weighted = x_p * h_p
    x_summed = x_weighted.sum(axis=1)
    return x_summed.T

def riallto_stuff(app, x, win_coeffs, M, P, W, dt):
    x = x[:int(len(x)//(M*P))*M*P] # Ensure it's an integer multiple of win_coeffs
    output = np.empty([M*W, P])
    
    for i in range(0, M*W-M):
        # Apply frontend, take FFT, then take power (i.e. square)
        sample_win = x[i*W:i*W+M*P]

        gg = runApp(app, M, P, sample_win, win_coeffs, dt)
        
        # x_fir = pfb_fir_frontend(sample_win, win_coeffs, M, P)

        x_pfb = fft(gg, P)

        x_psd = squaring_pfb(x_pfb)

        output[i, :] = x_psd
    

    return output

def pfb_spectrometer(app, x, win_coeffs, n_taps, n_chan, n_win, n_int, dt):
    M = n_taps
    P = n_chan
    W = n_win
    
    start = time.time()
    x_psd = riallto_stuff(app, x, win_coeffs, M, P, W, dt)
    time_passed = time.time() - start
    print("Time for Riallto stuff: " + str(time_passed) + "s")
    # Trim array so we can do time integration
    x_psd = x_psd[:np.round(x_psd.shape[0]//n_int)*n_int]
    
    # Integrate over time, by reshaping and summing over axis (efficient)
    x_psd = x_psd.reshape(x_psd.shape[0]//n_int, n_int, x_psd.shape[1])
    x_psd = x_psd.mean(axis=1)

    print("DONE!")
    return x_psd