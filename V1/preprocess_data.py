from datetime import datetime
import sys
import os
import torch
from ecg_h5_dataset import ECGGenerationDataset
from paths import *


class PreprocessData:
    def __init__(
        self,
        mode: str,
    ):
        self._mode = mode
        self._samples_per_patient = 2 if self._mode == 'Val' else config["samples_per_patient"]
        self._window_size = 128*5
        self._ecg_channels = 2
        self._ecg_sample_segments_num = int(23040/self._window_size)
        
        directory = '/mnt/qnap/yonatane/data/ltafdb'
        all_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
        all_files.remove('/mnt/qnap/yonatane/data/ltafdb/30.h5')
                
        self._num_patients = int(0.7*len(all_files)) if mode == 'Train' or mode == 'Val' else len(all_files)-int(0.7*len(all_files))
        start_index = 0 if mode == 'Train' or mode == 'Val' else int(0.7*len(all_files))
        
        self._ecgs = torch.empty(self._num_patients, self._ecg_channels, self._ecg_sample_segments_num*self._samples_per_patient, self._window_size)
        
        ecgs_all_files = [os.path.join(directory, f) for f in all_files[start_index : start_index+self._num_patients]]

        for idx, f in enumerate(ecgs_all_files):
            ecg = ECGGenerationDataset(record_path=f, mode='Train', window_size=self._window_size)
            patient_data = self.__get_patient_data(ecg)
            self._ecgs[idx,:,:,:] = patient_data
            print('Patient data extracted: ', idx)

        
    def __get_patient_data(self, ecg):
        start_index = len(ecg) - self._samples_per_patient
        X_patient_data = torch.empty(self._ecg_channels, self._ecg_sample_segments_num*self._samples_per_patient, self._window_size)

        for i, j in zip(range(start_index, len(ecg)), range(len(ecg)-start_index)):
            assert len(ecg[i]['x'][0]) == len(ecg[i]['x'][1])
            X_patient_data[:, j*self._ecg_sample_segments_num:(j+1)*self._ecg_sample_segments_num,:] = ecg[i]['x']
        
        return X_patient_data
    
    
    def get_ecgs(self):
        predictions = config["forecast_predictions"]
        X_data = torch.empty(self._ecg_channels, self._samples_per_patient*self._ecg_sample_segments_num*self._num_patients, self._window_size)
        row = 0
        for i in range(0, self._ecgs.shape[2], predictions):
            for j in range(self._ecgs.shape[0]):
                X_data[:, row:row+predictions, :] = self._ecgs[j,:,i:i+predictions,:] 
                row += predictions
                
        return X_data
    

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Timestamp:", timestamp)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    #pre-process data
    X_train_filename = TRAIN_DATA
    if X_train_filename.is_file():
        # file exists
        X_train_data = torch.load(X_train_filename).permute(1,0,2).contiguous()

    else:
        # create a file
        pre_data = PreprocessData('Train')
        X_train_data = pre_data.get_ecgs()
        torch.save(X_train_data, str(X_train_filename))

    X_test_filename = TEST_DATA
    if X_test_filename.is_file():
        # file exists
        X_test_data = torch.load(X_test_filename).permute(1,0,2).contiguous()

    else:
        # create a file
        pre_data = PreprocessData('Test')
        X_test_data = pre_data.get_ecgs()
        torch.save(X_test_data, str(X_test_filename))
        
    print(X_train_data.shape)
    print(X_test_data.shape)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Timestamp:", timestamp) 


if __name__ == '__main__':
    main()