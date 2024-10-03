import numpy as np

def pfb_fir_frontend(x, win_coeffs, M, P):
    W = x.shape[0] // M // P
    x_p = x.reshape((W*M, P)).T
    h_p = win_coeffs.reshape((M, P)).T
    x_summed = np.zeros((P, M * W - M + 1))
    for t in range(0, M*W-M + 1):
        x_weighted = x_p[:, t:t+M] * h_p
        x_summed[:, t] = x_weighted.sum(axis=1)
    return x_summed.T

def fft(x_p, P, axis=1):
    return np.fft.fft(x_p, P, axis=axis)

def pfb_filterbank(x, win_coeffs, M, P):
    x = x[:int(len(x)//(M*P))*M*P] # Ensure it's an integer multiple of win_coeffs
    x_fir = pfb_fir_frontend(x, win_coeffs, M, P)
    x_pfb = fft(x_fir, P)
    return x_pfb

def squaring_pfb(x_pfb):
    x_psd = np.real(x_pfb * np.conj(x_pfb)) # same as x_psd = np.abs(x_pfb)**2
    return x_psd

def riallto_stuff(x, win_coeffs, M, P):
    # Apply frontend, take FFT, then take power (i.e. square)
    x_pfb = pfb_filterbank(x, win_coeffs, M, P)
    x_psd = squaring_pfb(x_pfb)
    return x_psd

def pfb_spectrometer(x, n_taps, n_chan, n_int, win_coeffs):
    M = n_taps
    P = n_chan
    
    x_psd = riallto_stuff(x, win_coeffs, M, P)
    
    # Trim array so we can do time integration
    x_psd = x_psd[:np.round(x_psd.shape[0]//n_int)*n_int]
    
    # Integrate over time, by reshaping and summing over axis (efficient)
    x_psd = x_psd.reshape(x_psd.shape[0]//n_int, n_int, x_psd.shape[1])
    x_psd = x_psd.mean(axis=1)
    
    return x_psd