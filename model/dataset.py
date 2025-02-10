import numpy as np
import scipy.signal as signal
import pyedflib
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import torch
import os

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, data, scoring_len):
        # get data and preprocess
        pass

    def __len__(self):
        # return length of the data
        pass

    def __getitem__(self, index):
        # return data at index
        pass

def load_dataset(data_dir, batch_size):
    pass

def preprocess(edf_file):
    with pyedflib.EdfReader(edf_file) as f:
        num_signals = f.signals_in_file
        print(f"Number of signals: {num_signals}")

        original_fs = f.getSampleFrequency(0)
        eeg_signal = f.readSignal(0)
        print(f"EEG signal shape: {eeg_signal.shape}")
        print(f"Length of EEG signal: {len(eeg_signal)}")

    # resample to 256 Hz
    print("Resampling from: ", original_fs, " to 256 Hz")
    target_fs = 256
    num_samples = int(len(eeg_signal) * (target_fs / original_fs))
    eeg_signal_resampled = signal.resample(eeg_signal, num_samples)
    print(f"Resampled EEG signal shape: {eeg_signal_resampled.shape}")

    # short time fourier transform (STFT) with hamming windows?
    frame_size = 256    # Corresponds to 2 seconds
    step_size = 16      # Step size for overlapping windows
    fs = 1000           # Sampling frequency
    window = signal.windows.hamming(frame_size)

    # compute STFT manually
    frequencies, times, Zxx = signal.stft(eeg_signal_resampled, fs=fs, window=window, nperseg=frame_size, noverlap=frame_size-step_size)
    print(f"Shape of frequencies: {frequencies.shape}")
    # compute PSD
    psd = np.abs(Zxx) ** 2  # Estimate as squared magnitude of FFT

    # band pass (0.5 - 24 Hz)
    freq_mask = (frequencies >= 0.5) & (frequencies <= 24)
    filtered_psd = psd[freq_mask, :]
    print(f"Shape of psd: {psd.shape}")
    print(f"Shape of freq mask: {freq_mask.shape}")
    print(f"Shape of filtered_psd: {filtered_psd.shape}")

    log_psd = np.log1p(filtered_psd)

    # standardize frequency component (Zero Mean, Unit Variance)
    standardized_psd = (log_psd - np.mean(log_psd, axis=1, keepdims=True)) / np.std(log_psd, axis=1, keepdims=True)

    # visualize
    print(f"Shape of times: {times.shape}")
    print(f"Shape of standardized_psd: {standardized_psd.shape}")
    start_time = 0
    end_time = 2
    start = np.searchsorted(times, start_time)
    end = np.searchsorted(times, end_time)

    times_subset = times[start:end]
    standardized_psd_subset = standardized_psd[:, start:end]

    print(f"frequencies: {frequencies}")
    print(f"frequencies masked: {frequencies[freq_mask]}")

    # plt.figure(figsize=(10, 5))
    # plt.pcolormesh(times_subset, frequencies[freq_mask], standardized_psd_subset, shading='gouraud', cmap='jet')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Standardized EEG Spectrogram')
    # plt.colorbar(label='')
    # plt.show()

root_dir = os.getcwd()
edf_dirA = root_dir + '/training_data/CohortA/recordings/'
edf_dirB = root_dir + '/training_data/CohortB/recordings/'
edf_dirC = root_dir + '/training_data/CohortC/recordings/'
edf_dirD = root_dir + '/training_data/CohortD/recordings/'
if __name__ == '__main__':
    # input length = scoring length * 1024
    # 1382401 / 10800 / 32 = 4 second epochs
    edf_file = edf_dirC + 'C2.edf'
    print("Processing: ", edf_file)

    preprocess(edf_file)