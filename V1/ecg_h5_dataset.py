import torch
from torch import from_numpy, Tensor, float32
from torch.utils.data import Dataset
from typing import Dict
import sys
import matplotlib.pyplot as plt


import h5py
import numpy as np


class ECGGenerationDataset(Dataset):
    """
    A Dataset class for generating samples suitable for training a generative model for generating windows of ECG leads
    """

    def __init__(
            self,
            record_path: str,
            mode: str,
            prediction_horizon: int = 1,
            samples_overlap: int = 0,
            sample_length: int = 128 * 60 * 3,
            window_size: int = 128 * 60,
            train_ratio: float = 0.6,
            val_ratio: float = 0.15,
            buffer_size: int = 1024,
            leads_as_channels: bool = False,
            n_dims: int = 4,
    ):
        super().__init__()

        assert train_ratio + val_ratio < 1
        assert mode in ('Train', 'Val', 'Test')
        assert sample_length % window_size == 0, f"Can't divide {sample_length} points to {window_size}-long windows"

        self._record_path = record_path
        self._mode = mode
        self._prediction_horizon = prediction_horizon
        self._samples_overlap = samples_overlap
        self._sample_length = sample_length
        self._window_size = window_size
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._test_ratio = 1 - train_ratio - val_ratio
        self._buffer_size = buffer_size
        self._leads_as_channels = leads_as_channels
        self._step = sample_length - samples_overlap
        self._n_dims = n_dims

        with h5py.File(record_path, mode='r') as f:
            data = f['/record']
            x_shape = data['x'].shape
            y_shape = data['y'].shape
            rpeaks_shape = data['rpeaks'].shape
            last_rpeak = data['rpeaks'][-1]

            assert y_shape == rpeaks_shape, f"{record_path} does not have a label for each annotated beat!"
            assert last_rpeak < x_shape[0], f"Latest beat annotation is not part of the signal for {record_path}!"
            self._total_length = ((x_shape[0] - sample_length) // (sample_length - samples_overlap)) - prediction_horizon # we made total_length a member

        self._train_length = int(self._total_length * train_ratio)
        self._val_length = int(self._total_length * val_ratio)
        self._test_length = int(self._total_length * self._test_ratio)

        if mode == 'Train':
            self._length = self._train_length
            self._start_index = 0

        elif mode == 'Val':
            self._length = self._val_length
            self._start_index = self._train_length

        else:
            self._length = self._test_length
            self._start_index = self._train_length + self._val_length

    def __len__(self) -> int:
        return self._total_length # we return all samples not only Train, changed from self._length to self._total_length

    def __getitem__(
            self,
            index: int,
    ) -> Dict[str, Tensor]:
        rpeaks_buffer = -np.ones(self._buffer_size)
        labels_buffer = -np.ones(self._buffer_size)
        effective_index = (index * self._step) + self._start_index
        with h5py.File(self._record_path, mode='r') as hdf5_file:
            record = hdf5_file['/record']
            x = record['x'][effective_index:(effective_index + self._sample_length)]
            x = np.swapaxes(x, 0, 1)

            if self._n_dims == 4:
                x = np.reshape(x, (x.shape[0], -1, self._window_size))

            if self._prediction_horizon > 0:
                y = [
                    record['x'][
                    (effective_index + i * self._sample_length):(effective_index + ((i + 1) * self._sample_length))
                    ]
                    for i in range(self._prediction_horizon)
                ]
                y = np.concatenate(y, 0)
                y = np.swapaxes(y, 0, 1)

                if self._n_dims == 4:
                    y = np.reshape(y, (y.shape[0], -1, self._window_size))

            else:
                y = x.copy()

            rpeaks = record['rpeaks'][:]
            first_rpeak = np.where(rpeaks >= effective_index)[0]
            if len(first_rpeak):
                first_rpeak = first_rpeak[0]

            else:
                first_rpeak = None

            last_rpeak = np.where(rpeaks < (effective_index + self._sample_length))[0]
            if len(last_rpeak):
                last_rpeak = last_rpeak[-1]

            else:
                last_rpeak = None

            if first_rpeak is None or last_rpeak is None:
                rpeaks = rpeaks_buffer
                labels = labels_buffer

            else:
                rpeaks = record['rpeaks'][first_rpeak:last_rpeak + 1]
                labels = record['y'][first_rpeak:last_rpeak + 1]

        rpeaks_buffer[:len(rpeaks)] = rpeaks
        labels_buffer[:len(labels)] = labels

        sample = {
            'x': from_numpy(x).type(float32),
            'y': from_numpy(y).type(float32),
            'rpeaks': from_numpy(rpeaks_buffer).type(float32),
            'labels': from_numpy(labels_buffer).type(float32),
        }

        return sample

def main():
    ecg = ECGGenerationDataset(record_path='/mnt/qnap/yonatane/data/ltafdb/08.h5', mode='Test', window_size=128)
    print(type(ecg))
    start_index = len(ecg) - 220
    X_patient_data = torch.tensor([])
    for i in range(start_index, len(ecg)):
        assert len(ecg[i]['x'][0]) == len(ecg[i]['x'][1])
        print(ecg[i]['x'][0].size())
        for j in range(len(ecg[i]['x'][0])):
            X_patient_data = torch.cat((X_patient_data, torch.cat((ecg[i]['x'][0][j], ecg[i]['x'][1][j])).unsqueeze(0)))
            print(ecg[i]['x'][0][j].size())
            print(X_patient_data.size())
            print(j)

            plt.figure(figsize=(10, 4))  # Set the figure size (optional)
            plt.plot(X_patient_data[j].numpy(), label='ECG Lead')  # Plot the ECG data; convert tensor to numpy array if needed
            plt.xlabel('Time Points')  # Set the x-axis label
            plt.ylabel('Amplitude')  # Set the y-axis label
            plt.title(f'ECG Lead at index {i}, segment {j}')  # Set the title
            plt.legend()  # Add a legend (optional)
            plt.show()  # Display the plot

            if j == 1:
                sys.exit()
            
if __name__ == '__main__':
    main()