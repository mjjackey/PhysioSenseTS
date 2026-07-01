import pickle

def load_subject_data(data_path, subject_id):
    """
    Load data for a specific subject.  
    :param subject_id: The ID of the subject to load data for."""  
    # Implement the logic to load data for the specified subject
    # For example, you might read data from a file or database
    try:
        with open(f"{data_path}/{subject_id}/{subject_id}.pkl", 'rb') as f:
            data = pickle.load(f,encoding="latin1")
            print(f"Loaded data for subject {subject_id}")
            return data
    except FileNotFoundError as e:
        print(f"Error:{e}")
        return None