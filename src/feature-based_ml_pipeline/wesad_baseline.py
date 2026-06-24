import pickle
from matplotlib import streamplot
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.stats import mode

E4_BVP_SAMPLING_RATE = 64 # Empatica E4 BVP sensor sampling rate(Hz)
E4_EDA_TEMP_SAMPLING_RATE = 4 # Empatica E4 EDA and temperature sensor sampling rate(Hz)
E4_ACC_SAMPLING_RATE = 32 # Empatica E4 ACC sensor sampling rate(Hz)
E4_WINDOW_SIZE_SEC = 60
E4_WINDOW_STEP_SEC = 30 # overlap of 30 seconds

class WESADPipeline:
    def __init__(self):
       self.model = None
       self.feature_names = None
    
    def load_subject_data(self, subject_id):
        """
        Load the subject data from a file.
        Args: subject_id (int): The subject ID.
        Returns: A pandas DataFrame containing the subject data.
        """
        DATA_PATH_PREFIX='/mnt/d/DataSets/WESAD'
        try:
           with open(f"{DATA_PATH_PREFIX}/{subject_id}/{subject_id}.pkl", 'rb') as f:
              data = pickle.load(f,encoding="latin1")
              print(f"Loaded data for subject {subject_id}")
              return data
        except FileNotFoundError as e:
           print(f"Error:{e}")
           return None
    

    def create_window(self, data, type:str, window_size, step_size):
        """
        Create a window of data from the subject data.
        Args: data (pandas DataFrame): The subject E4 wrist sonsor dat
        type (str): The type of data to extract. 'BVP', 'EDA', 'TEMP', 'ACC'.
        window_size (int): The nubmer of samples in the window 
        step_size (int): The number of samples of step. 
        Returns: A numpy array containing the windowed data.
        """
        windowed_data = []
        num_steps = int(len(data)//step_size) + 1
        for i in range(num_steps):
            start = i * step_size
            end = min(start + window_size, len(data))
            # for BVP: the data is R peaks index array
            if type == 'BVP':
                windowed_data.append(data[data>start & data<end])
            else:
                windowed_data.append(data[start:end])
        return np.array(windowed_data)

    def e4_bvp_extraction(self, data):
        """
        Extract the BVP data of Empatica E4 sensor from the subject data.
        Args: data (pandas DataFrame): The subject data.
        Returns: A pandas DataFrame containing the E4 BVP HRV features data.
        """
        bvp_data = data['BVP']
        _,bvp_info=nk.ppg_process(bvp_data, sampling_rate= E4_BVP_SAMPLING_RATE)
        bvp_R_peaks_indices = bvp_info['PPG_Peaks']
        # Period convert to milliseconds
        bvp_T = 1/E4_BVP_SAMPLING_RATE*1000  
        bvp_rr_intervals=np.diff(bvp_R_peaks_indices)*bvp_T
        bvp_rr_events=bvp_R_peaks_indices[1:]*bvp_T
        bvp_rr_df = pd.DataFrame({'RR_Intervals(ms)': bvp_rr_intervals, 'RR_Event(ms)': bvp_rr_events})
        print(f"Extracted {len(bvp_rr_events)} RR intervals.")
        print(bvp_rr_df[:5])
        
        wined_data_np = self.create_window(bvp_R_peaks_indices,'BVP',E4_WINDOW_SIZE_SEC*E4_BVP_SAMPLING_RATE,E4_WINDOW_STEP_SEC*E4_BVP_SAMPLING_RATE)
        df_columns =['RR_Mean','SDNN','RMSSD']
        df_rows = wined_data_np.shape[0]
        hrv_matrix = np.zeros((df_rows, len(df_columns)))
        index = ['W'+str(i+1) for i in range(df_rows)]
        hrv_df = pd.DataFrame(hrv_matrix, index=index, columns=df_columns)
        for i in range(wined_data_np.shape[0]):
            wined_rr = np.diff(wined_data_np[i])*bvp_T
            print(f"Window {i+1} first 5 RR intervals: \n {wined_rr[:5]}")
            hrv = nk.hrv_time(wined_data_np[i], E4_BVP_SAMPLING_RATE)
            hrv_row = hrv.iloc[0].to_dict()
            print(f"HRV features for window {i+1}: \n 'HRV_MeanNN':{hrv_row['HRV_MeanNN']}, 'HRV_SDNN': {hrv_row['HRV_SDNN']}, 'HRV_RMSSD': {hrv_row['HRV_RMSSD']}")
            hrv_df.loc['W'+str(i+1),'RR_Mean'] = hrv_row['HRV_MeanNN']
            hrv_df.loc['W'+str(i+1),'SDNN'] = hrv_row['HRV_SDNN']
            hrv_df.loc['W'+str(i+1),'RMSSD'] = hrv_row['HRV_RMSSD']
        return hrv_df

    def e4_eda_extraction(self, data):
        """
        Extract the EDA data of Empatica E4 sensor from the subject data.
        Args: data (pandas DataFrame): The subject data.
        Returns: A numpy array containing the E4 EDA data.
        """
        e4_eda_pd, e4_eda_info = nk.eda_process(data.ravel(),E4_EDA_TEMP_SAMPLING_RATE)
        print("SCR Count:", e4_eda_pd['SCR_Peaks'].sum())
        print(e4_eda_pd.loc[e4_eda_pd['SCR_Peaks']==1, ["SCR_Amplitude"]])


    def e4_temp_extraction(self, data):
        """
        Extract the temperature data of Empatica E4 sensor from the subject data.
        Args: data (pandas DataFrame): The subject data.
        Returns: A numpy array containing the E4 temperature data.
        """
        pass

    def e4_acc_extraction(self, data):
        """
        Extract the ACC data of Empatica E4 sensor from the subject data.
        Args: data (pandas DataFrame): The subject data.
        Returns: A numpy array containing the E4 ACC data.
        """
        pass

    

    
    def feature_extraction(self, data):
        """
        Extract features of from the subject data.
        Args: data (pandas DataFrame): The subject data.
        Returns: A numpy array containing the extracted features.
        """
        # Extract features from the data
        # For example, you can extract the mean, standard deviation, and correlation of each feature
        features = data.iloc[:, 1:].values
        return features
    
    def assign_labels(self, data):
        """
        Assign labels to the subject data.
        Args: data (pandas DataFrame): The subject data.
        Returns: A numpy array containing the assigned labels.
        """
        # Assign labels to the data
        # For example, you can assign the label '1' to the data of the subject who is in the resting state
        labels = np.zeros(len(data))
        labels[data['Resting State'] == 1] = 1
        return labels
    
    def build_subject_dataset(self, subject_id):
        """
        Build the subject dataset from the subject data.
        Args: subject_id (int): The subject ID.
        Returns: A pandas DataFrame containing the subject dataset.
        """
        data = self.load_subject_data(subject_id)
        if data is None:
            return None             
        bvp_data = self.e4_bvp_extraction(data)
        eda_data = self.e4_eda_extraction(data)
        temp_data = self.e4_temp_extraction(data)
        
    def bulid_all_subject_datasets(self):
        """
        Build all the subject datasets from the subject data.
        Returns: A list of pandas DataFrames containing the subject datasets.
        """
        subject_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        subject_datasets = []
        for subject_id in subject_ids:
            subject_dataset = self.build_subject_dataset(subject_id)

if __name__ == "__main__":
    pass