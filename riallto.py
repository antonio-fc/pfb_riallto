import numpy as np

def pfb_fir_frontend(x, win_coeffs, M, P, W):
    x_p = x.reshape((W*M, P)).T
    h_p = win_coeffs.reshape((M, P)).T
    x_summed = np.zeros((P, M * W)) # x_summed = np.zeros((P, M * W - M + 1)) ???
    for t in range(0, M*W-M + 1):
        x_weighted = x_p[:, t:t+M] * h_p
        x_summed[:, t] = x_weighted.sum(axis=1)
    return x_summed.T

def fft(x_p, P, axis=1):
    return np.fft.fft(x_p, P, axis=axis)

def squaring_pfb(x_pfb):
    return np.real(x_pfb * np.conj(x_pfb)) # same as x_psd = np.abs(x_pfb)**2

def riallto_stuff(x, win_coeffs, M, P, W):
    print("Input type:")
    x = x[:int(len(x)//(M*P))*M*P] # Ensure it's an integer multiple of win_coeffs
    print(x.shape)
    print(len(x))
    print(type(x))

    # Apply frontend, take FFT, then take power (i.e. square)
    print("\nFIR frontend:")
    x_fir = pfb_fir_frontend(x, win_coeffs, M, P, W)
    print(x_fir.shape)
    print(len(x_fir))
    print(type(x_fir))

    print("\nFFT:")
    x_pfb = fft(x_fir, P) # pfb filterbank
    print(x_pfb.shape)
    print(len(x_pfb))
    print(type(x_pfb))

    print("\nSquare (output):")
    x_psd = squaring_pfb(x_pfb)
    print(x_psd.shape)
    print(len(x_psd))
    print(type(x_psd))
    

    return x_psd

def pfb_spectrometer(x, n_taps, n_chan, n_win, n_int, win_coeffs):
    M = n_taps
    P = n_chan
    W = n_win
    
    x_psd = riallto_stuff(x, win_coeffs, M, P, W)
    
    # Trim array so we can do time integration
    x_psd = x_psd[:np.round(x_psd.shape[0]//n_int)*n_int]
    
    # Integrate over time, by reshaping and summing over axis (efficient)
    x_psd = x_psd.reshape(x_psd.shape[0]//n_int, n_int, x_psd.shape[1])
    x_psd = x_psd.mean(axis=1)
    
    return x_psd