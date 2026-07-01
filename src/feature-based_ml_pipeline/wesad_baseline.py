import os
import pickle
import sys
sys.path.append('/home/jackey/github/PhysioSenseTS')
from src.datasets.load_data import load_subject_data
from src.features.features_hd import save_features, load_features
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold

DATA_PATH_PREFIX='/home/jackey/datasets/WESAD'
E4_BVP_SAMPLING_RATE = 64 # Empatica E4 BVP sensor sampling rate(Hz)
E4_EDA_TEMP_SAMPLING_RATE = 4 # Empatica E4 EDA and temperature sensor sampling rate(Hz)
E4_ACC_SAMPLING_RATE = 32 # Empatica E4 ACC sensor sampling rate(Hz)
E4_WINDOW_SIZE_SEC = 60
E4_WINDOW_STEP_SEC = 30 # overlap of 30 seconds
LABEL_SAMPLING_RATE = 700 # Lable sampling rate(Hz)
FEATURES_PATH_PREFIX = '/home/jackey/github/PhysioSenseTS/data/processed'

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
        # num_steps = len(data)//step_size + 1
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
        self.num_steps = len(bvp_data)//(E4_WINDOW_STEP_SEC*E4_BVP_SAMPLING_RATE) + 1
        _,bvp_info=nk.ppg_process(bvp_data, sampling_rate= E4_BVP_SAMPLING_RATE)
        bvp_R_peaks_indices = bvp_info['PPG_Peaks']
        # Period convert to milliseconds
        bvp_T = 1/E4_BVP_SAMPLING_RATE*1000  
        bvp_rr_intervals=np.diff(bvp_R_peaks_indices)*bvp_T
        bvp_rr_events=bvp_R_peaks_indices[1:]*bvp_T
        bvp_rr_df = pd.DataFrame({'RR_Intervals(ms)': bvp_rr_intervals, 'RR_Event(ms)': bvp_rr_events})
        print(f"Extracted {len(bvp_rr_events)} RR intervals.")
        print(bvp_rr_df[:5])
        
        # bvp_wined_list = self.create_window(bvp_R_peaks_indices,'BVP',E4_WINDOW_SIZE_SEC*E4_BVP_SAMPLING_RATE,E4_WINDOW_STEP_SEC*E4_BVP_SAMPLING_RATE)
        df_columns =['Start_time','End_time','RR_Mean','SDNN','RMSSD']
        df_rows = self.num_steps
        hrv_matrix = np.zeros((df_rows, len(df_columns)))
        self.index = ['W'+str(i+1) for i in range(df_rows)]
        hrv_df = pd.DataFrame(hrv_matrix, index=self.index, columns=df_columns)
        for i in range(self.num_steps):
            start = i * E4_WINDOW_STEP_SEC * E4_BVP_SAMPLING_RATE
            end = start + E4_WINDOW_SIZE_SEC * E4_BVP_SAMPLING_RATE 
            bvp_wined = bvp_R_peaks_indices[(bvp_R_peaks_indices>start) & (bvp_R_peaks_indices<end)]
            wined_rr = np.diff(bvp_wined)*bvp_T
            # print(f"Window {i+1} first 5 RR intervals: \n {wined_rr[:5]}")
            hrv = nk.hrv_time(bvp_wined, E4_BVP_SAMPLING_RATE)
            hrv_row = hrv.iloc[0].to_dict()
            # print(f"HRV features for window {i+1}: \n 'HRV_MeanNN':{hrv_row['HRV_MeanNN']}, 'HRV_SDNN': {hrv_row['HRV_SDNN']}, 'HRV_RMSSD': {hrv_row['HRV_RMSSD']}")
            hrv_df.loc['W'+str(i+1),df_columns[0]] = i * E4_WINDOW_STEP_SEC
            hrv_df.loc['W'+str(i+1),df_columns[1]] = i * E4_WINDOW_STEP_SEC + E4_WINDOW_SIZE_SEC
            hrv_df.loc['W'+str(i+1),df_columns[2]] = hrv_row['HRV_MeanNN']
            hrv_df.loc['W'+str(i+1),df_columns[3]] = hrv_row['HRV_SDNN']
            hrv_df.loc['W'+str(i+1),df_columns[4]] = hrv_row['HRV_RMSSD']
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
        
        df_columns =['EDA_Tonic_mean','EDA_Phasic_mean','SCR_Count','SCR_Mean_Amplitude']
        df_rows = self.num_steps
        eda_matrix = np.zeros((df_rows, len(df_columns)))
        eda_feature_df = pd.DataFrame(eda_matrix, index=self.index, columns=df_columns)
        for i in range(self.num_steps):
            start = i * E4_WINDOW_STEP_SEC * E4_EDA_TEMP_SAMPLING_RATE
            end = start + E4_WINDOW_SIZE_SEC * E4_EDA_TEMP_SAMPLING_RATE
            e4_eda_win = e4_eda_pd_df.iloc[start:end]
            
            tonic_mean = e4_eda_win['EDA_Tonic'].mean()
            eda_feature_df.loc['W'+str(i+1),df_columns[0]]=tonic_mean

            phasic_mean = e4_eda_win['EDA_Phasic'].mean()
            eda_feature_df.loc['W'+str(i+1),df_columns[1]]=phasic_mean

            scr_peaks_count = e4_eda_win['SCR_Peaks'].sum()
            eda_feature_df.loc['W'+str(i+1),df_columns[2]]=scr_peaks_count   

            scr_amplitude_mean = e4_eda_win['SCR_Amplitude'].mean()
            eda_feature_df.loc['W'+str(i+1),df_columns[3]]=scr_amplitude_mean
        return eda_feature_df

    def e4_temp_extraction(self, data):
        """
        Extract the temperature data of Empatica E4 sensor from the subject data.
        Args: data (Numpy array): The subject E4 sensor data.
        Returns: A pandas DataFrame containing the E4 temperature features data.
        """
        e4_temp = data['TEMP']
        # temp_wined_np = np.array(self.create_window(e4_temp,'TEMP',E4_WINDOW_SIZE_SEC*E4_EDA_TEMP_SAMPLING_RATE,E4_WINDOW_STEP_SEC*E4_EDA_TEMP_SAMPLING_RATE))
        df_columns =['Temp_Mean','Temp_Std','Temp_Slope']
        df_rows = self.num_steps
        temp_matrix = np.zeros((df_rows, len(df_columns)))
        temp_feature_df = pd.DataFrame(temp_matrix, index=self.index, columns=df_columns)
        for i in range(self.num_steps):
            start = i * E4_WINDOW_STEP_SEC * E4_EDA_TEMP_SAMPLING_RATE
            end = start + E4_WINDOW_SIZE_SEC * E4_EDA_TEMP_SAMPLING_RATE
            e4_temp_win = e4_temp[start:end]
            temp_mean = e4_temp_win.mean()
            temp_std = e4_temp_win.std()
            temp_slope = np.polyfit(range(len(e4_temp_win)), e4_temp_win, 1)[0]
            temp_feature_df.loc['W'+str(i+1),df_columns[0]]=temp_mean
            temp_feature_df.loc['W'+str(i+1),df_columns[1]]=temp_std
            temp_feature_df.loc['W'+str(i+1),df_columns[2]]=temp_slope
        return temp_feature_df


    def e4_acc_extraction(self, data):
        """
        Extract the ACC data of Empatica E4 sensor from the subject data.
        Args: data (Numpy array): The subject E4 sensor data.
        Returns: A pandas DataFrame containing the E4 ACC features data.
        """
        e4_acc = data['ACC']
        # acc_wined_np = np.array(self.create_window(e4_acc,'ACC',E4_WINDOW_SIZE_SEC*E4_ACC_SAMPLING_RATE,E4_WINDOW_STEP_SEC*E4_ACC_SAMPLING_RATE))
        df_columns =['ACC_Mean','ACC_Std','ACC_Eng']
        df_rows = self.num_steps
        acc_matrix = np.zeros((df_rows, len(df_columns)))
        acc_feature_df = pd.DataFrame(acc_matrix, index=self.index, columns=df_columns)
        for i in range(self.num_steps):
            start = i * E4_WINDOW_STEP_SEC * E4_ACC_SAMPLING_RATE
            end = start + E4_WINDOW_SIZE_SEC * E4_ACC_SAMPLING_RATE
            e4_acc_win = e4_acc[start:end]
            acc_mag = np.linalg.norm(e4_acc_win, axis=1)
            acc_mean = acc_mag.mean()
            acc_std = acc_mag.std()
            acc_eng = np.sum(acc_mag**2)
            acc_feature_df.loc['W'+str(i+1),df_columns[0]]=acc_mean
            acc_feature_df.loc['W'+str(i+1),df_columns[1]]=acc_std
            acc_feature_df.loc['W'+str(i+1),df_columns[2]]=acc_eng
        return acc_feature_df

    
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
    
    def filter_df_labels(self, feature_df):
        """
        Filter the labels of the subject data.
        Args: feature_df (pandas DataFrame): The subject features data.
        Returns: A pandas DataFrame containing the filtered labels.
        """
        feature_hd_df = feature_df[feature_df["Label"].isin([1,2,3])]
        feature_hd_df["Label"] = feature_hd_df["Label"].map(lambda x: x-1)
        print(feature_hd_df["Label"].value_counts())
        return feature_hd_df
    
    def build_subject_dataset(self, subject_id):
        """
        Build the subject dataset from the subject data.
        Args: subject_id (int): The subject ID.
        Returns: A pandas DataFrame containing the subject's features dataset.
        """
        # data = self.load_subject_data(subject_id)
        data = load_subject_data(DATA_PATH_PREFIX, subject_id)
        if data is None:
            return None  
        e4_wrist_data = data['signal']['wrist']
        # e4_wrist_data = e4_wrist_data.sort_values(by='timestamp', ascending=True)           
        bvp_df = self.e4_bvp_extraction(e4_wrist_data)
        eda_df = self.e4_eda_extraction(e4_wrist_data)
        temp_df = self.e4_temp_extraction(e4_wrist_data)
        acc_df = self.e4_acc_extraction(e4_wrist_data)
        labels = data['label']
        labels_pd_np = self.assign_labels(labels)
        labels_pd_df = pd.DataFrame(labels_pd_np,index=self.index, columns=['Label'])
        feature_df = pd.concat([bvp_df, eda_df, temp_df, acc_df, labels_pd_df], axis=1)
        feature_df['Subject_ID'] = subject_id
        save_features(feature_df, FEATURES_PATH_PREFIX, subject_id)  # Save the features to a file
        print(feature_df.head())
        return feature_df
     
    def build_all_subject_datasets(self):
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
                return None
            else:
                subject_datasets.append(sub_df)
        all_df = pd.concat(subject_datasets, axis=0,ignore_index=True)
        print(all_df.head())
        save_features(all_df, FEATURES_PATH_PREFIX, 'all')
        return all_df
    
    def train_test_split_sub(self,feature_df):
        """
        Build the train and test datasets for independent subject data or for all subjects data mixed together.
        Returns: A tuple containing the train and test datasets.
        """
        X = feature_df.drop(columns=['Subject_ID','Start_time','End_time','Label'])
        y = feature_df['Label']
        print("Final X shape:", X.shape)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        return X_scaled, y, X_train, X_test, y_train, y_test
    
    def predict(self, X_test):
        """
        Predict the labels of the test data using a trained Random Forest model.
        Returns: A Numpy array containing the predicted labels.
        """
        y_pred = self.model.predict(X_test)
        return y_pred
    
    def evaluation(self, X_scaled, X_test, y, y_test, y_pred):
        """
        Evaluate the performance of the trained Random Forest model.
        Returns: A classification report and confusion matrix.
        """
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2], yticklabels=[0,1,2])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        print("Test accuracy:", self.model.score(X_test, y_test))
        scores = cross_val_score(self.model, X_scaled, y, cv=5)
        print("Cross-validation accuracy: %.2f ± %.2f" % (scores.mean(), scores.std()))

    def train_and_evaluate(self,feature_df):
        """
        Train a Random Forest model on the subject data.
        Returns: A trained Random Forest model.
        """
        X = feature_df.drop(columns=['Subject_ID','Start_time','End_time','Label'])
        y = feature_df['Label']
        print("Final X shape:", X.shape)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        # print("X_scaled first 5 rows:\n", X_scaled[:5])
        # print("y first 5 elements:", y[:5])

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        self.model = rf_model
        self.feature_names = X.columns.tolist()
        y_pred = rf_model.predict(X_test)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2], yticklabels=[0,1,2])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        print("Test accuracy:", self.model.score(X_test, y_test))
        cv = KFold(n_splits=5, shuffle=False)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv)
        print("Cross-validation accuracy: %.2f ± %.2f" % (scores.mean(), scores.std()))

    def run_pipe_sub_indep(self, subject_id):
        """
        Run the feature-based ML pipeline for a single subject.
        Returns: None
        """
        # feature_df = self.build_subject_dataset(subject_id)
        feature_df = load_features(FEATURES_PATH_PREFIX, subject_id)  
        if feature_df is None:
            print(f"No data found for subject ID: {subject_id}")
            return
        feature_df_filtered = self.filter_df_labels(feature_df)
        self.train_and_evaluate(feature_df_filtered)

    def run_pipe_subs_mixed(self):
        """
        Run the feature-based ML pipeline for all subjects mixed together.
        Returns: None
        """
        # feature_df = self.build_all_subject_datasets()
        feature_df = load_features(FEATURES_PATH_PREFIX, 'all')  # Load the features for the first subject
        if feature_df is None:
            return
        feature_df_filtered = self.filter_df_labels(feature_df)
        self.train_and_evaluate(feature_df_filtered)

if __name__ == "__main__":
    pipeline = WESADPipeline()
    SUBJECT_ID = 'S2'
    # pipeline.build_subject_dataset(SUBJECT_ID)
    pipeline.run_pipe_sub_indep(SUBJECT_ID)
    # pipeline.build_all_subject_datasets()
    
    