import numpy as np
import scipy.signal as signal
import pyedflib
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
import torch
import os
import pandas as pd

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
    psd = np.abs(Zxx) ** 2  # Estimate as squared magnitude of FFT

    # print(f"Shape of frequencies: {frequencies.shape}")
    return psd, frequencies, times

def preprocess(eeg_signal, original_fs, is_emg=False):
    # resample to 128 Hz
    target_fs = 128
    eeg_signal_resampled = eeg_signal
    if(original_fs != target_fs):
        print("Resampling to to 128Hz")
        num_samples = int(len(eeg_signal) * (target_fs / original_fs))
        eeg_signal_resampled = signal.resample(eeg_signal, num_samples)
        print(f"Resampled EEG signal shape: {eeg_signal_resampled.shape}")
    
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
    print(f"Final shape {standardized_psd.shape}")

    # plot(times, frequencies[freq_mask], standardized_psd)
    return standardized_psd

def get_scorings(scoring_file):
    scorings = pd.read_csv(scoring_file, header=None, usecols=[1], names=['scorings'])
    return scorings

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
class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, data_dirs, label_dirs):
        # get data and preprocess
        self.data = np.empty((48, 0), dtype=np.float32)
        for edf_dir in data_dirs:
            print("Processing directory: ", edf_dir)
            for filename in sorted(os.listdir(edf_dir)):
                if not filename.endswith('.edf'):
                    raise ValueError("File is not an EDF file")
                edf_file = os.path.join(edf_dir, filename)
                print("Processing: ", edf_file)
                with pyedflib.EdfReader(edf_file) as f:
                    assert f.signals_in_file == 3, "Unexpected number of signals.. there should be 3."
                    for i in range (3):
                        original_fs = f.getSampleFrequency(i)
                        eeg_signal = f.readSignal(i)
                        # print(f"EEG signal shape: {eeg_signal.shape}")
                        # print(f"Length of EEG signal: {len(eeg_signal)}")
                        # print(f"Original frequency: {original_fs}")
                        processed_data = preprocess(eeg_signal, original_fs, is_emg=(i == 2))
                        self.data = np.concatenate((self.data, np.array(processed_data)), axis=1)
            print("")
        print(f"Length of data: {self.data.shape}")
        print(f"Shape of data: {self.data.shape}")

        self.all_scorings = []
        for scoring_dir in label_dirs:
            for filename in sorted(os.listdir(scoring_dir)):
                if not filename.endswith('.csv'):
                    raise ValueError("File is not a CSV file")
                scoring_file = os.path.join(scoring_dir, filename)
                print("Processing: ", scoring_file)
                scorings = get_scorings(scoring_file)
                # print(scorings[21590:])
                assert len(scorings) == 21600
                self.all_scorings.extend(scorings['scorings'])
        
        self.all_scorings = np.array(self.all_scorings, dtype=np.str_)
        print(f"Shape of scorings: {self.all_scorings.shape}")


    def __len__(self):
        return len(self.all_scorings)

    def __getitem__(self, index):
        start_idx = index * 32
        end_idx = start_idx + 32
        
        # slice data for 48x32 item -> we will need 5 of these
        data_item = self.data[:, start_idx:end_idx]
        label_item = self.all_scorings[index]
        return data_item, label_item

def load_dataset(data_dir, batch_size):
    pass

if __name__ == '__main__':
    dataset = EEGDataset(edf_dirs, scoring_dirs)

    # for cohort A and B @ 256 Hz hmmm
    # 1382401 / 10800 / 32 = 4 second epochs
    # for cohort C and D
    # 691201 / 21600 = 32
    # edf_file = edf_dirA + "A1.edf"
    # with pyedflib.EdfReader(edf_file) as f:
    #     eeg_signal = f.readSignal(2)
    #     original_fs = f.getSampleFrequency(2)
    #     print(f"EEG signal shape: {eeg_signal.shape}")
    #     print(f"original frequency: {original_fs}")
    #     preprocess(eeg_signal, original_fs, is_emg=True)

    # for edf_dir in edf_dirs:
    #     print("Processing directory: ", edf_dir)
    #     for filename in sorted(os.listdir(edf_dir)):
    #         if not filename.endswith('.edf'):
    #             raise ValueError("File is not an EDF file")
    #         edf_file = os.path.join(edf_dir, filename)
    #         print("Processing: ", edf_file)
    #         with pyedflib.EdfReader(edf_file) as f:
    #             num_signals = f.signals_in_file
    #             # print(f"Number of signals: {num_signals}")
    #             for i in range (3):
    #                 original_fs = f.getSampleFrequency(i)
    #                 eeg_signal = f.readSignal(i)
    #                 # print(f"EEG signal shape: {eeg_signal.shape}")
    #                 # print(f"Length of EEG signal: {len(eeg_signal)}")
    #                 # print(f"Original frequency: {original_fs}")
    #                 preprocess(eeg_signal, original_fs, is_emg=(i == 2))
    #     print("")

    # all_scorings = []
    # for scoring_dir in scoring_dirs:
    #     for filename in sorted(os.listdir(scoring_dir)):
    #         if not filename.endswith('.csv'):
    #             raise ValueError("File is not a CSV file")
    #         scoring_file = os.path.join(scoring_dir, filename)
    #         print("Processing: ", scoring_file)
    #         scorings = get_scorings(scoring_file)
    #         # print(scorings[21590:])
    #         assert len(scorings) == 21600
    #         all_scorings.append(scorings)
    # print(len(all_scorings))