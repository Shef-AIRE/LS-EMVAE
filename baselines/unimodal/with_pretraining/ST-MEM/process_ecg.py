import os
import numpy as np
import pandas as pd
import pickle as pkl
import pydicom

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, 'ECG_PAP')
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_ecg')
INDEX_PATH = os.path.join(BASE_DIR, 'index.csv')

# Parameters
TARGET_FS = 500  # Target sampling rate for all ECGs
TARGET_LENGTH = 6000  # Target sequence length (10 seconds)
NUM_LEADS = 12

def process_dicom(file_path):
    try:
        dicom_data = pydicom.dcmread(file_path)
        if "WaveformSequence" in dicom_data:
            rhythm_waveform = dicom_data.WaveformSequence[0]
            wave_data = rhythm_waveform.get("WaveformData")
            num_channels = rhythm_waveform.NumberOfWaveformChannels
            wave_array = np.frombuffer(wave_data, dtype=np.int16)

            if wave_array.size % num_channels == 0:
                wave_array = wave_array.reshape(-1, num_channels)
                if wave_array.shape[1] != NUM_LEADS:
                    print(f"Skipping {file_path} due to mismatched lead count.")
                    return None

                # Resample to the target length
                resampled_data = resample_ecg(wave_array, TARGET_LENGTH)
                # Normalize data
                normalized_data = (resampled_data - np.mean(resampled_data, axis=0)) / np.std(resampled_data, axis=0)
                return normalized_data
        return None
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def resample_ecg(ecg_data, target_length):
    from scipy.signal import resample
    return resample(ecg_data, target_length, axis=0)

def save_pkl(data, output_path):
    try:
        with open(output_path, 'wb') as file:
            pkl.dump(data, file)
    except Exception as e:
        print(f"Error saving file {output_path}: {e}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    index_entries = []

    for label_folder in ['Label_0', 'Label_1']:
        label = int(label_folder.split('_')[1])
        label_dir = os.path.join(INPUT_DIR, label_folder)

        for root, _, files in os.walk(label_dir):
            for file_name in files:
                if file_name.lower().endswith('.dcm'):
                    file_path = os.path.join(root, file_name)
                    ecg_data = process_dicom(file_path)

                    if ecg_data is not None:
                        relative_output_path = os.path.relpath(file_path, INPUT_DIR).replace(os.sep, '_')
                        pkl_file_name = f"{relative_output_path}.pkl"
                        pkl_output_path = os.path.join(OUTPUT_DIR, pkl_file_name)
                        save_pkl(ecg_data, pkl_output_path)

                        index_entries.append({'FILE_NAME': pkl_file_name, 'SAMPLE_RATE': TARGET_FS, 'LABEL': label})

    # Save the index CSV file
    pd.DataFrame(index_entries).to_csv(INDEX_PATH, index=False, columns=['FILE_NAME', 'SAMPLE_RATE', 'LABEL'])
    print(f"Index saved to {INDEX_PATH}")

if __name__ == "__main__":
    main()
