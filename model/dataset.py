import numpy as np
import scipy.signal as signal
import pyedflib
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import torch
import os
import pandas as pd
from tqdm import tqdm

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
    plt.title('Spectrogram')
    plt.colorbar(label='')
    plt.show()

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
    psd = np.abs(Zxx) ** 2  # Estimate as squared magnitude of FFT

    # print(f"Shape of frequencies: {frequencies.shape}")
    return psd, frequencies, times

def preprocess(eeg_signal, original_fs, is_emg=False):
    # resample to 128 Hz
    target_fs = 128
    eeg_signal_resampled = eeg_signal
    if(original_fs != target_fs):
        # print("Resampling to to 128Hz")
        num_samples = int(len(eeg_signal) * (target_fs / original_fs))
        eeg_signal_resampled = signal.resample(eeg_signal, num_samples)
        # print(f"Resampled EEG signal shape: {eeg_signal_resampled.shape}")
    
    # psd, frequencies, times = fast_fourier_transform(eeg_signal_resampled)
    psd, frequencies, times = short_fourier_transform(eeg_signal_resampled)

    if (is_emg == False):
        # band pass (0.5 - 24 Hz)
        freq_mask = (frequencies >= 0.5) & (frequencies <= 24)
        filtered_psd = psd[freq_mask, :]
        # print(frequencies[freq_mask])
        # print(f"Shape of times: {times.shape}")
        # print(f"Shape of psd: {psd.shape}")
        # print(f"Shape of freq mask: {freq_mask.shape}")
        # print(f"Shape of filtered_psd: {filtered_psd.shape}")
    else:
        # print("EMG signal detected")
        freq_mask = (frequencies >= 0.5) & (frequencies <= 24)
        emg_energy = np.sum(psd[freq_mask, :], axis=0)
        emg_energy_repeated = np.tile(emg_energy, (len(frequencies[freq_mask]), 1)) # Repeat the signal to form a consistent input for CNN
        # print(f"Shape of EMG energy: {emg_energy.shape}")
        # print(f"Shape of EMG energy repeated: {emg_energy_repeated.shape}")
        filtered_psd = emg_energy_repeated

    # standardize log frequency component (Zero Mean, Unit Variance)
    log_psd = np.log1p(filtered_psd)
    standardized_psd = (log_psd - np.mean(log_psd, axis=1, keepdims=True)) / np.std(log_psd, axis=1, keepdims=True)
    # print(f"Final shape {standardized_psd.shape}")

    # plot(times, frequencies[freq_mask], standardized_psd)
    return standardized_psd

def get_scorings(scoring_file):
    # NOTE: second column of lines 21581 - 21600 in B2.csv used to be ' now replaced with 1
    scorings = pd.read_csv(scoring_file, header=None, usecols=[2], names=['scorings'])
    scorings['scorings'] = scorings['scorings'].replace({'w': 1, 'n': 2, 'r': 3, "'": 1})
    scorings = scorings.infer_objects(copy=False)  # Explicitly infer object types, addresses warning
    scorings['scorings'] = scorings['scorings'].astype(int)
    scorings['scorings'] = scorings['scorings'].replace({1: 0, 2: 1, 3: 2})
    return scorings

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, data_dirs, label_dirs):
        self.offset = 2

        # get data and preprocess
        stacked_signals_list = []
        for edf_dir in data_dirs:
            # print("Processing directory: ", edf_dir)
            for filename in tqdm(sorted(os.listdir(edf_dir)), desc= f"Processing {edf_dir}"):
                if not filename.endswith('.edf'):
                    raise ValueError("File is not an EDF file")
                edf_file = os.path.join(edf_dir, filename)
                # print("Processing: ", edf_file)
                with pyedflib.EdfReader(edf_file) as f:
                    num_signals = f.signals_in_file
                    assert num_signals == 3, "Unexpected number of signals.. there should be 3."
                    signal = []
                    for i in range (num_signals):
                        original_fs = f.getSampleFrequency(i)
                        eeg_signal = f.readSignal(i)
                        processed_data = preprocess(eeg_signal, original_fs, is_emg=(i == 2))
                        # print(f"Processed data shape: {processed_data.shape}")
                        signal.append(processed_data)
                    stacked_signals = np.stack(signal, axis=0)
                    stacked_signals_list.append(stacked_signals)
            print("")
        self.data = np.concatenate(stacked_signals_list, axis=-1)
        # print(f"Shape of data: {self.data.shape}")

        # W -> 1, N -> 2, R -> 3
        self.all_scorings = []
        for scoring_dir in label_dirs:
            for filename in sorted(os.listdir(scoring_dir)):
                if not filename.endswith('.csv'):
                    raise ValueError("File is not a CSV file")
                scoring_file = os.path.join(scoring_dir, filename)
                # print("Processing: ", scoring_file)
                scorings = get_scorings(scoring_file)
                assert len(scorings) == 21600
                self.all_scorings.extend(scorings['scorings'])
        
        self.all_scorings = np.array(self.all_scorings, dtype=np.int_)
        # print(f"Shape of scorings: {self.all_scorings.shape}")


    def __len__(self):
        return len(self.all_scorings) - (2 * self.offset)

    def __getitem__(self, index):
        # NOTE: for each sample we will be taking 5 windows of 32x48
        # so for sample t, we will be taking t-2, t-1, t, t+1, t+2
        # start_idx doesn't take into account offset to avoid edge case at index 0, 1, n-2, n-1
        # the label index will take into account offset; i.e. label index starts at 2
        window_size = 32

        start_idx = index * window_size
        end_idx = start_idx + (5 * window_size)
        data_item = self.data[:, :, start_idx:end_idx]
        
        label_index = index + self.offset
        label_item = self.all_scorings[label_index]

        return data_item, label_item

if __name__ == '__main__':
    root_dir = os.getcwd()
    edf_dirA = root_dir + '/training_data/CohortA/recordings/'
    edf_dirB = root_dir + '/training_data/CohortB/recordings/'
    edf_dirC = root_dir + '/training_data/CohortC/recordings/'
    edf_dirD = root_dir + '/training_data/CohortD/recordings/'
    edf_dirs = [edf_dirA]
    scoring_dirA = root_dir + '/training_data/CohortA/scorings/'
    scoring_dirB = root_dir + '/training_data/CohortB/scorings/'
    scoring_dirC = root_dir + '/training_data/CohortC/scorings/'
    scoring_dirD = root_dir + '/training_data/CohortD/scorings/'
    scoring_dirs = [scoring_dirA]

    dataset = EEGDataset(edf_dirs, scoring_dirs)
    datapoint, label = dataset.__getitem__(0)
    print(datapoint.shape, label)
    print(f"Length of dataset: {dataset.__len__()}")
