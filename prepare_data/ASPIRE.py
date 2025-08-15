import os
import shutil
import os
import torch
import numpy as np
import wfdb
import neurokit2 as nk
from tqdm import tqdm
import gc
from collections import defaultdict
import pydicom

# Ensure output diectory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Monitor memory usage
def print_memory_usage():
    memory_used = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Memory usage: {memory_used:.2f} GB")


def preprocess_and_save_leads(ecg_root, output_dir):
    ensure_dir(output_dir)

    lead_names = [
        "LEAD_I", "LEAD_II", "LEAD_III", "LEAD_aVR",
        "LEAD_aVF", "LEAD_aVL", "LEAD_V1", "LEAD_V2",
        "LEAD_V3", "LEAD_V4", "LEAD_V5", "LEAD_V6"
    ]

    lead_data = {lead: [] for lead in lead_names}
    labels = []
    skipped_samples = 0

    # Process files for both labels
    for label_dir_name, label in [("Label_0", 0), ("Label_1", 1)]:
        label_dir = os.path.join(ecg_root, label_dir_name)
        if not os.path.exists(label_dir):
            print(f"Warning: {label_dir} does not exist. Skipping...")
            continue

        for folder_name in os.listdir(label_dir):
            subfolder_path = os.path.join(label_dir, folder_name, "1", "DICOM")
            if not os.path.exists(subfolder_path) or not os.listdir(subfolder_path):
                print(f"Empty or missing folder: {subfolder_path}. Skipping.")
                skipped_samples += 1
                continue

            dicom_file = os.listdir(subfolder_path)[0]  # Assume only one file per folder
            file_path = os.path.join(subfolder_path, dicom_file)

            try:
                # Process DICOM waveform data
                ecg_waveform = process_ecg_signal(file_path)

                # Validate and process each lead
                valid_lead_signals = []
                # Process and validate each lead
                for i, lead_name in enumerate(lead_names):
                    lead_signal = ecg_waveform[i]

                    # Interpolate missing values
                    lead_signal = nk.signal_interpolate(lead_signal, method="linear")

                    # Clean the ECG signal
                    lead_signal = nk.ecg_clean(lead_signal, sampling_rate=500, method="neurokit")

                    # Standardize the signal
                    lead_signal = nk.standardize(lead_signal)

                    # Check for invalid signals (NaNs or constant values)
                    if np.any(np.isnan(lead_signal)) or np.all(lead_signal == lead_signal[0]):
                        print(f"Invalid signal detected in {lead_name}. Skipping sample.")
                        print(f"Invalid sample path: {file_path}")  # Log the invalid file path here
                        valid_lead_signals = []
                        break

                    valid_lead_signals.append(lead_signal)

                if valid_lead_signals:
                    for lead_name, signal in zip(lead_names, valid_lead_signals):
                        lead_data[lead_name].append(signal)
                    labels.append(label)
                else:
                    skipped_samples += 1

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                skipped_samples += 1

    # Save each lead's data as .pt files
    print(f"\nTotal valid samples: {len(labels)}")
    print(f"Total skipped samples: {skipped_samples}")

    # Save leads and labels
    print("\nSaving leads and labels...")
    for lead_name, signals in lead_data.items():
        lead_tensor = torch.tensor(np.array(signals), dtype=torch.float32)
        torch.save(lead_tensor, os.path.join(output_dir, f"{lead_name}.pt"))
        print(f"Saved {lead_name} with shape {lead_tensor.shape}")

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    torch.save(labels_tensor, os.path.join(output_dir, "labels.pt"))
    print("Labels saved.")

    gc.collect()
    print_memory_usage()

def process_ecg_signal(file_path):
    """Read and extract ECG waveform data from a DICOM file."""
    dicom_data = pydicom.dcmread(file_path)
    if "WaveformSequence" not in dicom_data:
        raise ValueError(f"No WaveformSequence found in {file_path}. Invalid DICOM file.")

    rhythm_waveform = dicom_data.WaveformSequence[0]
    wave_data = rhythm_waveform.WaveformData
    num_channels = rhythm_waveform.NumberOfWaveformChannels
    wave_array = np.frombuffer(wave_data, dtype=np.int16).reshape(-1, num_channels)

    # Trim or pad waveform to 10 seconds (5000 samples per lead)
    desired_length = 5000
    if wave_array.shape[0] > desired_length:
        wave_array = wave_array[:desired_length, :]
    elif wave_array.shape[0] < desired_length:
        padding = np.zeros((desired_length - wave_array.shape[0], num_channels), dtype=wave_array.dtype)
        wave_array = np.vstack((wave_array, padding))

    return wave_array.T  # Return waveform with shape [12, 5000]


# Change here for PAWP
ecg_root = "E:/MICCAI_25_Finetune_dataset/ECG_PAP"
output_directory = "data_aspire_PAP_1/"

# Run preprocessing and save leads
preprocess_and_save_leads(ecg_root, output_directory)
