import pandas as pd
import torch
import numpy as np
import wfdb
from tqdm import tqdm
import neurokit2 as nk
import os
import psutil
import gc  # Garbage collector

# Ensure the output directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Monitor memory usage
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_used = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    print(f"Memory usage: {memory_used:.2f} GB")

# Preprocess and save each lead individually without redundant storage
def preprocess_and_save_leads(df, base_path, output_dir):
    full_paths = base_path + df['path']  # Correctly construct paths
    ensure_dir(output_dir)

    lead_names = [
        "LEAD_I", "LEAD_II", "LEAD_III", "LEAD_aVR",
        "LEAD_aVF", "LEAD_aVL", "LEAD_V1", "LEAD_V2",
        "LEAD_V3", "LEAD_V4", "LEAD_V5", "LEAD_V6"
    ]

    # Initialize lead-wise storage (empty lists for each lead)
    lead_data = {lead: [] for lead in lead_names}
    skipped_samples = 0

    # Process all files and directly store data in lead-wise structure
    print("Processing ECG data and skipping invalid samples...")
    for f in tqdm(full_paths, desc="Processing ECG data"):
        try:
            # Get the folder path
            folder_path = os.path.dirname(f)

            # Check if folder is empty
            if not os.path.exists(folder_path) or not os.listdir(folder_path):
                print(f"Empty folder: {folder_path}. Skipping.")
                skipped_samples += 1
                continue

            # Load the ECG signal and metadata
            wave_array, meta = wfdb.rdsamp(f)

            if meta["n_sig"] != 12:
                print(f"Skipping file {f}: Not a 12-lead ECG.")
                skipped_samples += 1
                continue

            # Process and validate each lead
            valid_lead_signals = []
            all_leads_valid = True

            for i, lead_name in enumerate(lead_names):
                lead_signal = wave_array[:, i]

                # Interpolate missing values
                lead_signal = nk.signal_interpolate(lead_signal, method="linear")

                # Clean the ECG signal
                lead_signal = nk.ecg_clean(
                    lead_signal,
                    sampling_rate=500,  # Replace with your data's sampling rate
                    method="neurokit"
                )

                # Check for NaNs or constant signals
                if np.any(np.isnan(lead_signal)) or np.all(lead_signal == lead_signal[0]):
                    print(f"Invalid signal detected in {lead_name}. Skipping sample.")
                    all_leads_valid = False
                    break  # Skip all leads if one is invalid

                # Normalize if valid
                lead_signal = nk.standardize(lead_signal)
                valid_lead_signals.append(lead_signal)

            # Only append data if all leads are valid
            if all_leads_valid:
                for lead_name, lead_signal in zip(lead_names, valid_lead_signals):
                    lead_data[lead_name].append(lead_signal)
            else:
                skipped_samples += 1

        except Exception as e:
            print(f"Error processing file {f}: {e}")
            skipped_samples += 1

    print(f"\nTotal valid samples: {len(lead_data['LEAD_I'])}")
    print(f"Total skipped samples: {skipped_samples}")

    # Verify lead data consistency
    print("\nVerifying lead data consistency...")
    lead_sample_counts = {lead: len(signals) for lead, signals in lead_data.items()}

    # Print the number of samples for each lead
    for lead, count in lead_sample_counts.items():
        print(f"{lead}: {count} samples")

    # Check if all leads have the same number of samples
    if len(set(lead_sample_counts.values())) == 1:
        print("\nAll leads have the same number of samples.")
    else:
        print("\nMismatch in sample counts across leads!")
        print("Leads with differing sample counts:")
        for lead, count in lead_sample_counts.items():
            print(f"{lead}: {count} samples")

    # Save each lead separately
    print("\nSaving leads individually...")
    for lead_name, signals in lead_data.items():
        print(f"Saving {lead_name}...")
        try:
            lead_tensor = torch.tensor(np.array(signals), dtype=torch.float32)
            lead_path = os.path.join(output_dir, f"{lead_name}.pt")
            torch.save(lead_tensor, lead_path)
            print(f"Saved {lead_name} with shape {lead_tensor.shape}")
        except Exception as e:
            print(f"Error saving {lead_name}: {e}")

        # Clear memory for this lead and trigger garbage collection
        del signals
        gc.collect()
        print_memory_usage()

    print("\nProcessing and saving complete!")

# Define parameters
base_path = '/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
ec_data_df = pd.read_csv(base_path + 'record_list.csv')
output_directory = os.path.join(base_path, 'data_feature/12_lead_ecg/')

# Process and save leads
preprocess_and_save_leads(ec_data_df, base_path, output_directory)