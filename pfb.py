"""
# pfb.py

Simple implementation of a polyphase filterbank.
"""
import numpy as np
import scipy
from scipy.signal import firwin, freqz, lfilter
from riallto import pfb_spectrometer


def db(x):
    """ Convert linear value to dB value """
    return 10*np.log10(x)

def generate_win_coeffs(M, P, window_fn="hamming"):
    win_coeffs = scipy.signal.get_window(window_fn, M*P)
    sinc       = scipy.signal.firwin(M * P, cutoff=1.0/P, window="rectangular")
    win_coeffs *= sinc
    return win_coeffs

if __name__ == "__main__":
    import pylab as plt
    import seaborn as sns
    sns.set_style("white")
    
    M     = 4          # Number of taps
    P     = 1024       # Number of 'branches', also fft length
    W     = 1000       # Number of windows of length M*P in input time stream
    n_int = 10          # Number of time integrations on output data

    # Generate a test data steam
    samples = np.arange(M*P*W)
    noise   = np.random.normal(loc=0.5, scale=0.1, size=M*P*W) 
    freq    = 1
    amp     = 1
    cw_signal = amp * np.sin(samples * freq)
    data = noise + cw_signal

    # Generate window coefficients
    win_coeffs = generate_win_coeffs(M, P, "hamming")
    pg = np.sum(np.abs(win_coeffs)**2)
    win_coeffs /= pg**.5 # Normalize for processing gain
    
    X_psd = pfb_spectrometer(data, n_taps=M, n_chan=P, n_int=2, win_coeffs=win_coeffs)

    plt.imshow(db(X_psd), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel("Channel")
    plt.ylabel("Time")
    
    plt.show()
    
