import numpy as np
import pyedflib
import scipy
import mat73
import os
import h5py
import time
import matplotlib.pyplot as plt
# from validate import valid

root_dir = '/Users/vtang/Downloads/skule/Thesis/sleep_rat/'
mat_dir = 'PSG05-mat/'
edf_dir = 'PSG05-edf/'

def upsample(data, original_sampling_rate, target_sampling_rate):
    duration = data.shape[0] / original_sampling_rate
    original_time_points = np.linspace(0, duration, data.shape[0])
    target_time_points = np.linspace(0, duration, int(data.shape[0] * (target_sampling_rate / original_sampling_rate)))

    interpolator = scipy.interpolate.interp1d(original_time_points, data, axis=0, kind='linear')
    data_upsampled = interpolator(target_time_points)

    return data_upsampled

def convert(filename):
    # Load your MATLAB file
    mat_file = root_dir + mat_dir + filename +'.mat'
    try:
        mat_data = scipy.io.loadmat(mat_file)
        print(f'Loading {filename} as older matlab version by scipy.io.loadmat')
    except NotImplementedError:
        mat_data = mat73.loadmat(mat_file)
        print(f'Loading {filename} as v7.3 format by mat73')

    # Extract EEG, ACC, EMG channels
    EEGc = mat_data['EEGc']
    EEGo = mat_data['ch2']
    ACCX = mat_data['acc_x']
    EMG  = mat_data['ch3']

    # Upsample accelerometer data with linear interpolation
    print("Upsample accelerometer data with linear interpolation")
    print(EEGc.shape[0])
    print(EEGo.shape[0])
    print(ACCX.shape[0])
    ACCX_upsampled = upsample(ACCX, 100, 1000)
    print("ACCX upsampled shape:", ACCX_upsampled.shape)

    # Define start time (tstart)
    # tstart = 1.727979813692845e+09

    # Create an EDF file
    if not os.path.exists(edf_dir):
        os.makedirs(edf_dir)
    edf_file = root_dir + edf_dir + filename + '.edf'

    with pyedflib.EdfWriter(edf_file, 3, file_type=1) as f:
        # Channel names
        # channel_names = ['EEGo', 'EEGc']
        channel_names = ['EEGo', 'EEGc', 'ACCX']
        # channel_names = ['EEGo', 'EEGc', 'ACCX', 'EMG']
        # Sample frequency (assuming 1000 Hz; adjust according to your data)
        sample_frequency = 1000
        truncate_length = (len(EEGc) // sample_frequency) * sample_frequency    # EDF f.writeSamples will pad up to multiple of sampling rate by repeating the last element. So just truncate the last tail of ori .mat data here
        # Write each channel
        # for i, data in enumerate([EEGo, EEGc]):
        for i, data in enumerate([EEGo, EEGc, ACCX_upsampled]):
            f.setSignalHeader(i, {
                'label': channel_names[i],
                'dimension': 'mV',
                'sample_frequency': sample_frequency,
                'physical_max': 10, #6.5,             # np.max(data[:, 1]),
                'physical_min': -10,#-6.3898,          # np.min(data[:, 1]),
                'digital_max': 32767,
                'digital_min': -32767,
                'physical_dimension': 'mV',
                'reserved': ''
            })
        # Prepare data for writing: stack all channels
        # all_data = np.vstack([EEGo[:, 1][:truncate_length], EEGc[:, 1][:truncate_length]])
        all_data = np.vstack([EEGo[:, 1][:truncate_length], EEGc[:, 1][:truncate_length], ACCX_upsampled[:, 1][:truncate_length]])

        f.writeSamples(all_data, digital=False)

    # Sanity check
    with pyedflib.EdfReader(edf_file) as f:
        for i in range(f.signals_in_file):
            # print(f"Signal length", f.getNSamples()[i])  # signal length: 2802000 data.shape (2802000,)
            data = f.readSignal(i)[:EEGc.shape[0]]
            # print(data.shape)
    print(f"EDF file {filename} created successfully.")

def plot_edf (edf_file):
    edf_file = edf_dir + edf_file + '.edf'
    print(edf_file)
    with pyedflib.EdfReader(edf_file) as f:
        # Get the number of signals in the file
        num_signals = f.signals_in_file
        print(f"Number of signals: {num_signals}")

        # Create a figure for the plot
        plt.figure(figsize=(10, num_signals * 2))

        # Iterate over each signal and plot it on a different axis
        for i in range(num_signals):
            label = f.getLabel(i)
            data = f.readSignal(i)

            print(f"Signal {i} label: {label}")
            print(f"Signal {i} data shape: {data.shape}")

            # Create a subplot for each signal
            plt.subplot(num_signals, 1, i + 1)
            plt.plot(data, label=label)
            plt.title(label)
            plt.xlabel("Time?")
            plt.ylabel("Voltage?")
            plt.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    filenames = []
    for file in os.listdir(root_dir + mat_dir):
        if file.endswith('.mat'):
            mat_filename = os.path.splitext(file)[0]
            convert(mat_filename)
            filenames.append(mat_filename)
    
    # plot_edf('20241030-2eeg')

    # print(len(filenames))
    # filenames = []
    # for file in os.listdir(root_dir + edf_dir):
    #     if file.endswith('.edf'):
    #         edf_filename = root_dir + edf_dir + file
    #         # valid(edf_filename)
    #         filenames.append(edf_filename)
    # print(len(filenames))
