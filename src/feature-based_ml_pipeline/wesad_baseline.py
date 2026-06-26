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
LABEL_SAMPLING_RATE = 700 # Lable sampling rate(Hz)

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
        Args: data (a Numpy array): The subject E4 wrist sonsor data
        type (str): The type of data to extract. 'BVP', 'EDA', 'TEMP', 'ACC'.
        window_size (int): The nubmer of samples in the window 
        step_size (int): The number of samples of step. 
        Returns: A Numpy array containing the windowed data.
        """
        windowed_data = []
        # num_steps = int(len(data)//step_size) + 1
        for i in range(self.num_steps):
            start = i * step_size
            end = min(start + window_size, len(data))
            # for BVP: the data is R peaks index array
            if type == 'BVP':
                r_win = data[(data>start) & (data<end)]
                print(f"R peaks in window{i+1}: \n {r_win}")
                windowed_data.append(r_win.tolist())
            else:
                windowed_data.append(data[start:end].tolist())
        return windowed_data

    def e4_bvp_extraction(self, data):
        """
        Extract the BVP data of Empatica E4 sensor from the subject data.
        Args: data (Numpy array): The subject E4 sensor data.
        Returns: A pandas DataFrame containing the E4 BVP HRV features data.
        """
        bvp_data = data['BVP']
        self.num_steps = int(len(bvp_data)//(E4_WINDOW_STEP_SEC*E4_BVP_SAMPLING_RATE)) + 1
        _,bvp_info=nk.ppg_process(bvp_data, sampling_rate= E4_BVP_SAMPLING_RATE)
        bvp_R_peaks_indices = bvp_info['PPG_Peaks']
        # Period convert to milliseconds
        bvp_T = 1/E4_BVP_SAMPLING_RATE*1000  
        bvp_rr_intervals=np.diff(bvp_R_peaks_indices)*bvp_T
        bvp_rr_events=bvp_R_peaks_indices[1:]*bvp_T
        bvp_rr_df = pd.DataFrame({'RR_Intervals(ms)': bvp_rr_intervals, 'RR_Event(ms)': bvp_rr_events})
        print(f"Extracted {len(bvp_rr_events)} RR intervals.")
        print(bvp_rr_df[:5])
        
        bvp_wined_list = self.create_window(bvp_R_peaks_indices,'BVP',E4_WINDOW_SIZE_SEC*E4_BVP_SAMPLING_RATE,E4_WINDOW_STEP_SEC*E4_BVP_SAMPLING_RATE)
        df_columns =['RR_Mean','SDNN','RMSSD']
        df_rows = self.num_steps
        hrv_matrix = np.zeros((df_rows, len(df_columns)))
        self.index = ['W'+str(i+1) for i in range(df_rows)]
        hrv_df = pd.DataFrame(hrv_matrix, index=self.index, columns=df_columns)
        for i in range(self.num_steps):
            wined_rr = np.diff(bvp_wined_list[i])*bvp_T
            # print(f"Window {i+1} first 5 RR intervals: \n {wined_rr[:5]}")
            hrv = nk.hrv_time(np.array(bvp_wined_list[i]), E4_BVP_SAMPLING_RATE)
            hrv_row = hrv.iloc[0].to_dict()
            # print(f"HRV features for window {i+1}: \n 'HRV_MeanNN':{hrv_row['HRV_MeanNN']}, 'HRV_SDNN': {hrv_row['HRV_SDNN']}, 'HRV_RMSSD': {hrv_row['HRV_RMSSD']}")
            hrv_df.loc['W'+str(i+1),'RR_Mean'] = hrv_row['HRV_MeanNN']
            hrv_df.loc['W'+str(i+1),'SDNN'] = hrv_row['HRV_SDNN']
            hrv_df.loc['W'+str(i+1),'RMSSD'] = hrv_row['HRV_RMSSD']
        return hrv_df

    def e4_eda_extraction(self, data):
        """
        Extract the EDA data of Empatica E4 sensor from the subject data.
        Args: data (Numpy array): The subject E4 sensor data.
        Returns: A pandas DataFrame containing the E4 EDA features data.
        """
        e4_eda = data['EDA']
        e4_eda_pd_df, e4_eda_info = nk.eda_process(e4_eda.ravel(),E4_EDA_TEMP_SAMPLING_RATE)
        print("SCR Count:", e4_eda_pd_df['SCR_Peaks'].sum())
        print(e4_eda_pd_df.loc[e4_eda_pd_df['SCR_Peaks']==1, ["SCR_Amplitude"]])
        
        df_columns =['EDA_Tonic','EDA_Phasic','SCR_Count','SCR_Amplitude']
        df_rows = self.num_steps
        eda_matrix = np.zeros((df_rows, len(df_columns)))
        eda_feature_df = pd.DataFrame(eda_matrix, index=self.index, columns=df_columns)
        for i in range(self.num_steps):
            start = i * E4_WINDOW_STEP_SEC * E4_EDA_TEMP_SAMPLING_RATE
            end = start + E4_WINDOW_SIZE_SEC * E4_EDA_TEMP_SAMPLING_RATE
            e4_eda_win = e4_eda_pd_df.iloc[start:end]
            
            tonic_mean = e4_eda_win['EDA_Tonic'].mean()
            eda_feature_df.loc['W'+str(i),'EDA_Tonic_mean']=tonic_mean

            phasic_mean = e4_eda_win['EDA_Phasic'].mean()
            eda_feature_df.loc['W'+str(i),'EDA_Phasic_mean']=phasic_mean

            scr_peaks_count = e4_eda_win['SCR_Peaks'].sum()
            eda_feature_df.loc['W'+str(i),'SCR_Count']=scr_peaks_count   

            scr_amplitude_mean = e4_eda_win['SCR_Amplitude'].mean()
            eda_feature_df.loc['W'+str(i),'SCR_Mean_Amplitude']=scr_amplitude_mean
        return eda_feature_df

    def e4_temp_extraction(self, data):
        """
        Extract the temperature data of Empatica E4 sensor from the subject data.
        Args: data (Numpy array): The subject E4 sensor data.
        Returns: A pandas DataFrame containing the E4 temperature features data.
        """
        e4_temp = data['TEMP']
        temp_wined_np = np.array(self.create_window(e4_temp,'TEMP',E4_WINDOW_SIZE_SEC*E4_EDA_TEMP_SAMPLING_RATE,E4_WINDOW_STEP_SEC*E4_EDA_TEMP_SAMPLING_RATE))
        df_columns =['TEMP_Mean','TEMP_SD','Temp_Slope']
        df_rows = self.num_steps
        temp_matrix = np.zeros((df_rows, len(df_columns)))
        temp_feature_df = pd.DataFrame(temp_matrix, index=self.index, columns=df_columns)
        for i in range(self.num_steps):
            start = i * E4_WINDOW_STEP_SEC * E4_EDA_TEMP_SAMPLING_RATE
            end = start + E4_WINDOW_SIZE_SEC * E4_EDA_TEMP_SAMPLING_RATE
            e4_temp_win = temp_wined_np[start:end]
            temp_mean = e4_temp_win.mean()
            temp_std = e4_temp_win.std()
            temp_slope = np.polyfit(range(len(e4_temp_win)), e4_temp_win, 1)[0]
            temp_feature_df.loc['W'+str(i),'TEMP_Mean']=temp_mean
            temp_feature_df.loc['W'+str(i),'TEMP_SD']=temp_std
            temp_feature_df.loc['W'+str(i),'Temp_Slope']=temp_slope
        return temp_feature_df


    def e4_acc_extraction(self, data):
        """
        Extract the ACC data of Empatica E4 sensor from the subject data.
        Args: data (Numpy array): The subject E4 sensor data.
        Returns: A pandas DataFrame containing the E4 ACC features data.
        """
        e4_acc = data['ACC']
        acc_wined_np = np.array(self.create_window(e4_acc,'ACC',E4_WINDOW_SIZE_SEC*E4_ACC_SAMPLING_RATE,E4_WINDOW_STEP_SEC*E4_ACC_SAMPLING_RATE))
        df_columns =['ACC_Mean','ACC_Std','ACC_Eng']
        df_rows = self.num_steps
        acc_matrix = np.zeros((df_rows, len(df_columns)))
        acc_feature_df = pd.DataFrame(acc_matrix, index=self.index, columns=df_columns)
        for i in range(self.num_steps):
            start = i * E4_WINDOW_STEP_SEC * E4_ACC_SAMPLING_RATE
            end = start + E4_WINDOW_SIZE_SEC * E4_ACC_SAMPLING_RATE
            e4_acc_win = acc_wined_np[start:end]
            acc_mag = np.linalg.norm(e4_acc_win, axis=1)
            acc_mean = acc_mag.mean()
            acc_std = acc_mag.std()
            acc_eng = np.sum(acc_mag**2)
            acc_feature_df.loc['W'+str(i),'ACC_Mean']=acc_mean
            acc_feature_df.loc['W'+str(i),'ACC_Std']=acc_std
            acc_feature_df.loc['W'+str(i),'ACC_Eng']=acc_eng
        return acc_feature_df

    
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
    
    def inspect_label(self, labels):
        """
        Inspect the labels of the subject data. 
        Args: labels (Numpy array): The subject labels data.
        Returns: None
        """ 
        unique_labels = np.unique(labels)
        print(f"Unique labels: {unique_labels}")

        # Count the number of samples for each label
        lbl_cnt_dict={}
        lbl_name_dict={1: "Baseline", 2: "Stress", 3: "Amusement", 4: "Mediation", 0: "Transient"}
        for label in unique_labels:
            lbl_name = lbl_name_dict.get(label, "Unknown")
            lbl_cnt = len(labels[labels==label])
            period= lbl_cnt/LABEL_SAMPLING_RATE/60.0
            print(f"Label {label}({lbl_name:10}): {lbl_cnt:7,} samples, Duration: {period:.3f} minutes")
            lbl_cnt_dict[label] = (lbl_cnt,period)
    
    def assign_labels(self, labels):
        """
        Assign labels to the subject data.
        Args: data (Numpy array): The subject labels data.
        Returns: A Numpy array containing the assigned labels.
        """
        labels_pd=[]
        for i in range(self.num_steps):
            start_sec = i * E4_WINDOW_STEP_SEC
            end_sec = start_sec + E4_WINDOW_SIZE_SEC
            lab_win = labels[int(start_sec*LABEL_SAMPLING_RATE):int(end_sec*LABEL_SAMPLING_RATE)] 
            lab_win_mode = mode(lab_win,keepdims=False).mode
            labels_pd.append(lab_win_mode)
        return np.array(labels_pd)
    
    def build_subject_dataset(self, subject_id):
        """
        Build the subject dataset from the subject data.
        Args: subject_id (int): The subject ID.
        Returns: A pandas DataFrame containing the subject's features dataset.
        """
        data = self.load_subject_data(subject_id)
        e4_wrist_data = data['signal']['wrist']
        if data is None:
            return None             
        bvp_df = self.e4_bvp_extraction(e4_wrist_data)
        eda_df = self.e4_eda_extraction(e4_wrist_data)
        temp_df = self.e4_temp_extraction(e4_wrist_data)
        acc_df = self.e4_acc_extraction(e4_wrist_data)
        labels = data['label']
        labels_pd_np = self.assign_labels(labels)
        labels_pd_df = pd.DataFrame(labels_pd_np,index=self.index, columns=['Label'])
        feature_df = pd.concat([bvp_df, eda_df, temp_df, acc_df, labels_pd_df], axis=1)
        feature_df['Subject_ID'] = subject_id
        return feature_df
     
    def bulid_all_subject_datasets(self):
        """
        Build all the subject datasets from the subject data.
        Returns: A pandas DataFrame containing the subject datasets.
        """
        subject_ids = ['S'+ str(i) for i in range(2,18)]
        subject_datasets = []
        for subject_id in subject_ids:
            sub_df = self.build_subject_dataset(subject_id)
            if sub_df is None:
                print(f"No data found for subject ID: {subject_id}")
            else:
                subject_datasets.append(sub_df)
        all_df = pd.concat(subject_datasets, axis=0,ignore_index=True)
        return all_df


if __name__ == "__main__":
    pipeline = WESADPipeline()
    SUBJECT_ID = 'S2'
    feature_df = pipeline.build_subject_dataset(SUBJECT_ID)
    if feature_df is None:
        print("No data found for subject ID:", SUBJECT_ID)
    else:
        print(feature_df.head())
        # print(feature_df.describe())