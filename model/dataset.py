import numpy as np
import scipy.signal as signal
import pyedflib
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
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

def plot(times, frequency, standardized_psd):
    start_time = 0
    end_time = 2
    start = np.searchsorted(times, start_time)
    end = np.searchsorted(times, end_time)
    times_subset = times[start:end]
    standardized_psd_subset = standardized_psd[:, start:end]

    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times_subset, frequency, standardized_psd_subset, shading='gouraud', cmap='jet')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Standardized EEG Spectrogram')
    plt.colorbar(label='')
    plt.show()

def fast_fourier_transform(eeg_signal):
    frame_size = 256  # Frame size (2 seconds)
    step_size = 16  # Step size for overlap
    fs = 128  # Sampling frequency
    N = len(eeg_signal)  # Total number of samples
    num_frames = (N - frame_size) // step_size + 1

    frequencies = fftfreq(frame_size, d=1/fs)[:frame_size // 2]
    times = np.arange(num_frames) * (step_size / fs)

    # list to store FFT results
    Zxx = np.zeros((len(frequencies), num_frames), dtype=complex)
    window = np.hamming(frame_size)

    # Apply FFT on overlapping frames
    for i in range(num_frames):
        start = i * step_size
        end = start + frame_size
        frame = eeg_signal[start:end] * window  # Apply window
        fft_result = fft(frame)[:frame_size // 2]  # Compute FFT and keep positive frequencies
        Zxx[:, i] = fft_result  # Store result

    Zxx_magnitude = np.abs(Zxx)
    psd = Zxx_magnitude ** 2
    return psd, frequencies, times

def short_fourier_transform(eeg_signal):
    # short time fourier transform (STFT) with hamming windows?
    frame_size = 256    # Corresponds to 2 seconds
    step_size = 16      # Step size for overlapping windows
    fs = 128           # Sampling frequency
    window = signal.windows.hamming(frame_size)

    # compute STFT manually
    frequencies, times, Zxx = signal.stft(eeg_signal, fs=fs, window=window, nperseg=frame_size, noverlap=frame_size-step_size)
    # print(f"Shape of frequencies: {frequencies.shape}")
    # compute PSD
    psd = np.abs(Zxx) ** 2  # Estimate as squared magnitude of FFT
    return psd, frequencies, times

def preprocess(eeg_signal, original_fs):
    # resample to 128 Hz
    target_fs = 128
    eeg_signal_resampled = eeg_signal
    if(original_fs != target_fs):
        print("Resampling to to 128Hz")
        num_samples = int(len(eeg_signal) * (target_fs / original_fs))
        eeg_signal_resampled = signal.resample(eeg_signal, num_samples)
        print(f"Resampled EEG signal shape: {eeg_signal_resampled.shape}")
    
    # apply fourier
    # psd, frequencies, times = fast_fourier_transform(eeg_signal_resampled)
    psd, frequencies, times = short_fourier_transform(eeg_signal_resampled)

    # band pass (0.5 - 24 Hz)
    freq_mask = (frequencies >= 0.5) & (frequencies <= 24)
    filtered_psd = psd[freq_mask, :]
    # print(frequencies[freq_mask])
    # print(f"Shape of times: {times.shape}")
    # print(f"Shape of psd: {psd.shape}")
    # print(f"Shape of freq mask: {freq_mask.shape}")
    # print(f"Shape of filtered_psd: {filtered_psd.shape}")

    log_psd = np.log1p(filtered_psd)

    # standardize frequency component (Zero Mean, Unit Variance)
    standardized_psd = (log_psd - np.mean(log_psd, axis=1, keepdims=True)) / np.std(log_psd, axis=1, keepdims=True)
    print(f"Shape of standardized_psd: {standardized_psd.shape}")

    # plot(times, frequencies[freq_mask], standardized_psd)

root_dir = os.getcwd()
edf_dirA = root_dir + '/training_data/CohortA/recordings/'
edf_dirB = root_dir + '/training_data/CohortB/recordings/'
edf_dirC = root_dir + '/training_data/CohortC/recordings/'
edf_dirD = root_dir + '/training_data/CohortD/recordings/'
edf_dirs = [edf_dirA, edf_dirB, edf_dirC, edf_dirD]
if __name__ == '__main__':
    # for cohort A and B @ 256 Hz hmmm
    # 1382401 / 10800 / 32 = 4 second epochs
    # for cohort C and D
    # 691201 / 21600 = 32

    for edf_dir in edf_dirs:
        print("Processing directory: ", edf_dir)
        for filename in os.listdir(edf_dir):
            if not filename.endswith('.edf'):
                raise ValueError("File is not an EDF file")
            edf_file = os.path.join(edf_dir, filename)
            print("Processing: ", edf_file)
            with pyedflib.EdfReader(edf_file) as f:
                num_signals = f.signals_in_file
                # print(f"Number of signals: {num_signals}")
                for i in range (2):
                    original_fs = f.getSampleFrequency(i)
                    eeg_signal = f.readSignal(i)
                    # print(f"EEG signal shape: {eeg_signal.shape}")
                    # print(f"Length of EEG signal: {len(eeg_signal)}")
                    # print(f"Original frequency: {original_fs}")
                    preprocess(eeg_signal, original_fs)
        print("")