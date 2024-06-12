from datetime import datetime
import wfdb
import torch
import matplotlib.pyplot as plt
from collections import Counter
from paths import *

is_overfitting = config['is_overfitting']


class MITBIHDataProcessor:
    def __init__(self, directory='mitdb'):
        self._sample_per_second = 360
        self._input_signal_duration = config['input_signal_duration']
        self._forecast_predictions_mitbih = config['forecast_predictions']
        self._signals_per_patient_mitbih = config['signals_per_patient']
        self._samples_per_patient = self._signals_per_patient_mitbih*self._input_signal_duration*self._sample_per_second*self._forecast_predictions_mitbih
        self._forecast_signals = int(self._sample_per_second * self._input_signal_duration * self._forecast_predictions_mitbih)
        assert self._samples_per_patient <= 650000, \
            "Input signal duration is more than actual data available. Please reduce the duration."
        self._directory = directory
        self._patients_records = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
                                '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
                                '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
                                '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
                                '222', '223', '228', '230', '231', '232', '233', '234']
        
        # _patients_data is the processed data for patients, each value in the dictionary is of shape: [signals_per_patient_mitbih, 2, 360*input_signal_duration*forecast_predictions_mitbih]
        self._patients_data = {}

    def load_patients_data(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("Begin retrieving data for patients:", timestamp)

        patients_data = {}
        for record in self._patients_records:
            try:
                ecg_record = wfdb.rdrecord(record, pn_dir=self._directory, sampto=self._samples_per_patient)
                signal_array = ecg_record.p_signal
                num_rows = self._samples_per_patient // self._forecast_signals

                tensor_segments = []
                for i in range(num_rows):
                    start_index = i * self._forecast_signals
                    end_index = start_index + self._forecast_signals
                    segment = signal_array[start_index:end_index]
                    tensor_segments.append(torch.tensor(segment, dtype=torch.float32))

                patients_data[record] = torch.stack(tensor_segments).transpose(1, 2)
            except Exception as e:
                print(f"Failed to load record {record}: {e}")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("Finish retrieving data for patients:", timestamp)

        self._patients_data = patients_data
        return patients_data

    def prepare_model_input(self, mode='Train'):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Begin preparing model input for {mode}:", timestamp) 

        num_patients = len(self._patients_records)
        split_idx = int(num_patients * 0.7)

        if mode == 'Train':
            selected_records = self._patients_records[:split_idx]
        else:
            selected_records = self._patients_records[split_idx:]

        all_data = torch.empty(self._signals_per_patient_mitbih*len(selected_records), 2, self._forecast_signals)

        current_signal = 0
        idx = 0
        while True:
            done = True
            for record in selected_records:
                if record in self._patients_data:
                    data = self._patients_data[record]
                    if current_signal < data.shape[0]:
                        all_data[idx, :, :] = data[current_signal, :, :]
                        idx += 1
                        done = False
            if done:
                break
            current_signal += 1

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Finish preparing model input for {mode}:", timestamp) 

        return all_data

    def show_beat_counts(self, patient_data):
        for record, data in patient_data.items():
            try:
                annotations = wfdb.rdann(record, 'atr', pn_dir=self._directory)
                beat_counts = Counter(annotations.symbol)
                print(f"Record {record}: {beat_counts}")
            except Exception as e:
                print(f"Failed to load annotations for record {record}: {e}")

    def plot_beat_distribution(self, record): # not sure about this.. need to check
        try:
            annotations = wfdb.rdann(record, 'atr', pn_dir=self._directory)
            plt.figure(figsize=(10, 4))
            plt.hist(annotations.sample, bins=100, label=list(set(annotations.symbol)))
            plt.title(f'Beat Distribution for Record {record}')
            plt.xlabel('Time (sample index)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Failed to plot distribution for record {record}: {e}")

    # add method to the class to show live the data for a patient moving in time


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Begin preprocessing MITBIH timestamp:", timestamp)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    mitbih_dataset = MITBIHDataProcessor()
    if not is_overfitting:
        mitbih_dataset.load_patients_data()

    X_train_filename = TRAIN_DATA
    if X_train_filename.is_file():
        X_train_data = torch.load(str(X_train_filename))
    else:        
        if is_overfitting:
            print(f"Mode is overfitting, retrieving 1 signal to overfit") 
            X_train_data = torch.from_numpy(wfdb.rdrecord('100', pn_dir='mitdb', sampto=1080).p_signal).unsqueeze(0).permute(0, 2, 1).float()
            X_train_data = torch.cat([X_train_data, X_train_data, X_train_data, X_train_data], dim=0)
        else:
            X_train_data = mitbih_dataset.prepare_model_input('Train')
        torch.save(X_train_data, str(X_train_filename))

    X_test_filename = TEST_DATA
    if X_test_filename.is_file():
        X_test_data = torch.load(str(X_test_filename))
    else:
        if is_overfitting:
            X_test_data = torch.from_numpy(wfdb.rdrecord('100', pn_dir='mitdb', sampto=1080).p_signal).unsqueeze(0).permute(0, 2, 1).float()
            X_test_data = torch.cat([X_test_data, X_test_data, X_test_data, X_test_data], dim=0)
        else:
            X_test_data = mitbih_dataset.prepare_model_input('Test')
        torch.save(X_test_data, str(X_test_filename))
        
    print(X_train_data.shape)
    print(X_test_data.shape)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("End preprocessing MITBIH timestamp:", timestamp) 


if __name__ == '__main__':    
    main()
