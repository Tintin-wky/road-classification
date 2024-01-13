import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fft_analysis(time_signal, sampling_rate):
    # Perform the FFT on the time domain signal
    fft_signal = np.fft.fft(time_signal)
    fft_magnitude = np.abs(fft_signal) / len(time_signal)  # Normalize by the number of samples
    fft_phase = np.angle(fft_signal)

    # Generate frequency bins for the FFT output
    freqs = np.fft.fftfreq(len(time_signal), 1 / sampling_rate)

    # Only keep the positive half of the frequencies, magnitudes, and phases
    positive_indices = freqs > 0
    pos_freqs = freqs[positive_indices]
    pos_magnitude = fft_magnitude[positive_indices]
    pos_phase = fft_phase[positive_indices]

    return pos_freqs, pos_magnitude, pos_phase


class ImuSignal:
    def __init__(self, time_signal, sampling_rate=125):
        self.time_signal = np.array(time_signal)
        self.N = len(self.time_signal)
        self.sampling_rate = sampling_rate
        self.time = np.arange(self.N) / self.sampling_rate
        self.frequency, self.magnitude, _ = fft_analysis(self.time_signal, self.sampling_rate)
        self.U = len(self.frequency)
        self.features = self.calculate_all_features()

    def plot_time_domain(self):
        # 绘制时域信号
        plt.figure(figsize=(10, 4))
        plt.plot(self.time_signal, label="Time Domain Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("Time Domain Signal")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_frequency_domain(self):
        # 绘制频域信号
        plt.figure(figsize=(10, 4))
        plt.plot(self.frequency, self.magnitude, label="Frequency Domain Signal")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.title("Frequency Domain Signal")
        plt.legend()
        plt.grid()
        plt.show()

    def mean_amplitude(self):
        return np.mean(np.abs(self.time_signal))

    def square_root_amplitude(self):
        return (np.sum(np.sqrt(np.abs(self.time_signal))) / self.N) ** 2

    def max_value(self):
        return np.max(self.time_signal)

    def min_value(self):
        return np.min(self.time_signal)

    def peak(self):
        return self.max_value() - self.min_value()

    def peak_value(self):
        return np.max(np.abs(self.time_signal))

    def square_mean_root(self):
        return (np.sum(np.abs(self.time_signal) ** 2) / self.N) ** 2

    def root_mean_square(self):
        return np.sqrt(np.mean(self.time_signal ** 2))

    def crest_factor(self):
        return self.peak_value() / self.root_mean_square()

    def clearance_factor(self):
        return self.peak_value() / self.root_mean_square()

    def kurtosis(self):
        return np.sum((self.time_signal - self.mean_amplitude()) ** 4) / ((self.N - 1) * self.root_mean_square() ** 4)

    def variance(self):
        return np.var(self.time_signal)

    def standard_deviation(self):
        return np.std(self.time_signal)

    def skewness(self):
        return (np.sum((self.time_signal - self.mean_amplitude()) ** 3) / (
                (self.N - 1) * self.standard_deviation() ** 3))

    def waveform_factor(self):
        return self.root_mean_square() / self.mean_amplitude()

    def pulse_factor(self):
        return self.peak_value() / self.mean_amplitude()

    def residual_gap_factor(self):
        return self.max_value() / self.square_root_amplitude()

    def skewness_factor(self):
        return self.peak() / self.square_root_amplitude() ** 2

    def peak_factor(self):
        return self.kurtosis() / self.square_root_amplitude() ** 4

    def yield_factor(self):
        return self.peak() / self.square_root_amplitude()

    def mean_frequency(self):
        return np.mean(self.magnitude)

    def frequency_center(self):
        return np.sum(self.frequency * self.magnitude) / np.sum(self.magnitude)

    def variance_of_mean_frequency(self):
        return np.sum((self.magnitude - self.mean_frequency()) ** 2) / (self.U - 1)

    def median_frequency(self):
        return np.median(self.magnitude)

    def peak_frequency(self):
        return np.max(self.magnitude) - np.min(self.magnitude)

    def root_mean_square_frequency(self):
        return np.sqrt(np.sum(self.frequency ** 2 * self.magnitude) / np.sum(self.magnitude))

    def mean_square_frequency(self):
        return np.sum(self.frequency ** 2 * self.magnitude) / np.sum(self.magnitude)

    def root_mean_frequency_square(self):
        return np.sqrt(np.sum((self.magnitude - self.mean_frequency()) ** 2) / (self.U - 1))

    # Define a method to calculate all features and return as a dictionary
    def calculate_all_features(self):
        return {
            "Mean amplitude": self.mean_amplitude(),
            "Square root amplitude": self.square_root_amplitude(),
            "Maximum value": self.max_value(),
            "Minimum value": self.min_value(),
            "Peak": self.peak(),
            "Peak value": self.peak_value(),
            "Square mean root": self.square_mean_root(),
            "Root mean square": self.root_mean_square(),
            "Crest factor": self.crest_factor(),
            "Clearance factor": self.clearance_factor(),
            "Kurtosis": self.kurtosis(),
            "Variance": self.variance(),
            "Standard deviation": self.standard_deviation(),
            "Skewness": self.skewness(),
            "Waveform factor": self.waveform_factor(),
            "Pulse factor": self.pulse_factor(),
            "Residual factor": self.residual_gap_factor(),
            "Skewness factor": self.skewness_factor(),
            "Peak factor": self.peak_factor(),
            "Yield factor": self.yield_factor(),
            "Mean frequency": self.mean_frequency(),
            "Frequency center": self.frequency_center(),
            "Variance of mean frequency": self.variance_of_mean_frequency(),
            "Median frequency": self.median_frequency(),
            "Peak frequency": self.peak_frequency(),
            "Root mean square frequency": self.root_mean_square_frequency(),
            "Mean square frequency": self.mean_square_frequency(),
            "Root mean frequency square": self.root_mean_frequency_square()
        }


if __name__ == "__main__":
    csv_file = '../dataset/Bump/015_001.csv'
    df = pd.read_csv(csv_file)
    signal_data = df['angular_velocity_z']
    signal = ImuSignal(signal_data)
    signal.plot_time_domain()
    signal.plot_frequency_domain()
    print(signal.features)
