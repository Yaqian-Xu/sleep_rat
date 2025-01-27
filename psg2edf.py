import numpy as np
import pyedflib
import scipy
import mat73
import os
import h5py
import time
from validate import valid

root_dir = '/Users/xuyaqian/Documents/RatSleep/'
mat_dir = 'PSG05/'
edf_dir = 'PSG05_EDF_ocm/'


def convert(filename):
    # Load your MATLAB file
    mat_file = root_dir + mat_dir + filename +'.mat'
    try:
        mat_data = scipy.io.loadmat(mat_file)
        print(f'Loading {filename} as older matlab version by scipy.io.loadmat')
    except NotImplementedError:
        mat_data = mat73.loadmat(mat_file)
        print(f'Loading {filename} as v7.3 format by mat73')
        # with h5py.File(mat_file, 'r') as f:
        #     mat_data = {key: f[key][()] for key in f.keys()}
        #     print(f'Loading {filename} as v7.3 format by h5py')


    # Extract EEG and EMG channels
    EEGc = mat_data['EEGc']
    EEGo = mat_data['EEGo']   # xyq
    EMG = mat_data['EMG']
    assert EEGo.shape == EMG.shape
    assert EEGc.shape == EMG.shape
    print('EMG shape', EMG.shape)   # e.g., EMG shape (17775750, 2)

    # # Define start time (tstart)
    # tstart = 1.727979813692845e+09
    edf_file = root_dir + edf_dir + filename + '.edf'
    # Create an EDF file
    with pyedflib.EdfWriter(edf_file, 3, file_type=1) as f:           # num_channels: 3 xyq
        # Channel names
        channel_names = ['EEGo', 'EEGc', 'EMG']
        # Sample frequency (assuming 1000 Hz; adjust according to your data)
        sample_frequency = 1000
        truncate_length = (len(EEGc) // sample_frequency) * sample_frequency    # EDF f.writeSamples will pad up to multiple of sampling rate by repeating the last element. So just truncate the last tail of ori .mat data here
        # Write each channel
        for i, data in enumerate([EEGo, EEGc, EMG]):
            f.setSignalHeader(i, {
                'label': channel_names[i],
                'dimension': 'mV',
                'sample_rate': sample_frequency,
                'physical_max': 6.5,             # np.max(data[:, 1]),
                'physical_min': -6.3898,          # np.min(data[:, 1]),
                'digital_max': 32767,
                'digital_min': -32767,
                'physical_dimension': 'mV',
                'reserved': ''
            })
        # Prepare data for writing: stack all channels
        all_data = np.vstack([EEGo[:, 1][:truncate_length], EEGc[:, 1][:truncate_length], EMG[:, 1][:truncate_length]])
        print("all_data.shape", all_data.shape)

        # Write samples for all channels
        f.writeSamples(all_data, digital=False)
    with pyedflib.EdfReader(edf_file) as f:
        for i in range(f.signals_in_file):
            print(f"Signal length", f.getNSamples()[i])  # signal length: 2802000 data.shape (2802000,)

            # Read and output a portion of the signal data
            data = f.readSignal(i)[:EEGc.shape[0]]
            print(data.shape)
    print(f"EDF file {filename} created successfully.")


if __name__ == '__main__':
    filenames = []
    for file in os.listdir(root_dir + mat_dir):
        if file.endswith('.mat'):
            mat_filename = os.path.splitext(file)[0]
            convert(mat_filename)
            filenames.append(mat_filename)

    print(len(filenames))
    filenames = []
    for file in os.listdir(root_dir + edf_dir):
        if file.endswith('.edf'):
            edf_filename = root_dir + edf_dir + file
            valid(edf_filename)
            filenames.append(edf_filename)
    print(len(filenames))
