import numpy as np
from scipy.signal import butter, lfilter, spectrogram
import matplotlib.pyplot as plt

class GWSignalProcessing:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def butter_lowpass(self, cutoff, order=5):
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(self, data, cutoff, order=5):
        b, a = self.butter_lowpass(cutoff, order=order)
        y = lfilter(b, a, data)
        return y

    def compute_spectrogram(self, data):
        f, t, Sxx = spectrogram(data, self.sample_rate)
        return f, t, Sxx

    def plot_signal(self, data, title='Signal'):
        plt.figure(figsize=(10, 4))
        plt.plot(data)
        plt.title(title)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()

    def plot_spectrogram(self, f, t, Sxx):
        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.title('Spectrogram')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(label='Intensity [dB]')
        plt.show()

def main():
    sample_rate = 1024  # Sample rate in Hz
    gw_signal_processing = GWSignalProcessing(sample_rate)

    # Simulated gravitational wave signal (sine wave + noise)
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    frequency = 50  # Frequency of the signal
    signal = 0.5 * np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.normal(size=t.shape)

    # Plot original signal
    gw_signal_processing.plot_signal(signal, title='Original Signal')

    # Apply lowpass filter
    cutoff = 60  # Cutoff frequency in Hz
    filtered_signal = gw_signal_processing.lowpass_filter(signal, cutoff)

    # Plot filtered signal
    gw_signal_processing.plot_signal(filtered_signal, title='Filtered Signal')

    # Compute and plot spectrogram
    f, t, Sxx = gw_signal_processing.compute_spectrogram(filtered_signal)
    gw_signal_processing.plot_spectrogram(f, t, Sxx)

if __name__ == '__main__':
    main()
