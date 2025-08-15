import pandas as pd
import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from tqdm import tqdm
import neurokit2 as nk  # Ensure you have this installed: pip install neurokit2

# Load the dataset
df = pd.read_csv(r'D:/ukbiobank/combined_metadata_filtered.csv')

# Define required columns
cols_to_keep = [
    'Participant ID',
    'LV myocardial mass | Instance 2',
    'LA maximum volume | Instance 2'
]

# Filter and rename
df_filtered = df[cols_to_keep].copy()
df_filtered = df_filtered.dropna(subset=[
    'LV myocardial mass | Instance 2',
    'LA maximum volume | Instance 2'
])

# Rename columns
df_filtered.rename(columns={
    'LV myocardial mass | Instance 2': 'LVM',
    'LA maximum volume | Instance 2': 'LAV'
}, inplace=True)


# Calculate PAWP using the provided formula
df_filtered['PAWP'] = 6.1352 + (0.07204 * df_filtered['LAV']) + (0.02256 * df_filtered['LVM'])

# Optional: round PAWP values to 2 decimal places
df_filtered['PAWP'] = df_filtered['PAWP'].round(2)


# Create label column: 1 if PAWP > 15, else 0
df_filtered['label'] = (df_filtered['PAWP'] > 15).astype(int)

# Print confirmation and preview
print("Label column added based on PAWP > 15.")
print(df_filtered.head())

# Count how many 0s and 1s are in the label column
label_counts = df_filtered['label'].value_counts()


# Define paths
label_folders = {
    0: r"D:/ukbiobank/Label_0",
    1: r"D:/ukbiobank/Label_1"
}
output_dir = r"D:/ukbiobank/ECG_PAWP_UKB_Full"
os.makedirs(output_dir, exist_ok=True)

# 12-lead mapping
lead_map = {
    "I": "LEAD_I", "II": "LEAD_II", "III": "LEAD_III",
    "aVR": "LEAD_aVR", "aVL": "LEAD_aVL", "aVF": "LEAD_aVF",
    "V1": "LEAD_V1", "V2": "LEAD_V2", "V3": "LEAD_V3",
    "V4": "LEAD_V4", "V5": "LEAD_V5", "V6": "LEAD_V6"
}

# Storage
lead_data = {v: [] for v in lead_map.values()}
labels = []
skipped = 0

def load_strip_ecg_data(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        strip_element = root.find('.//StripData')
        if strip_element is None:
            return None

        resolution_element = strip_element.find('Resolution')
        if resolution_element is None:
            return None
        resolution = float(resolution_element.text)

        sample_rate_element = strip_element.find('SampleRate')
        sample_rate = float(sample_rate_element.text) if sample_rate_element is not None else 500.0

        channel_sample_count = int(strip_element.find('ChannelSampleCountTotal').text)
        if channel_sample_count < 5000:
            return None  # Not a full-length ECG

        ecg_data = {
            'signals': {},
            'sample_rate': sample_rate,
            'resolution': resolution
        }

        for waveform_element in strip_element.findall('WaveformData'):
            lead_name = waveform_element.get('lead')
            waveform_text = waveform_element.text
            if lead_name and waveform_text:
                raw_values = [int(val.strip()) for val in waveform_text.replace(',', ' ').split()]
                if len(raw_values) < 5000:
                    continue  # Incomplete signal

                signal = np.array(raw_values[:5000], dtype=np.float32)

                # Step 1: Interpolate missing values
                signal = nk.signal_interpolate(signal, method="linear")

                # Step 2: Clean ECG signal
                signal = nk.ecg_clean(signal, sampling_rate=500, method="neurokit")

                # Step 3: Standardize signal
                signal = nk.standardize(signal)

                ecg_data['signals'][lead_name] = signal

        return ecg_data

    except Exception:
        return None

# Parse and process files
for label, folder in label_folders.items():
    for fname in tqdm(os.listdir(folder), desc=f"Label {label}"):
        if not fname.endswith(".xml"):
            continue

        fpath = os.path.join(folder, fname)
        ecg_info = load_strip_ecg_data(fpath)

        if ecg_info is None or len(ecg_info['signals']) < 12:
            skipped += 1
            continue

        signal_dict = ecg_info['signals']
        current_sample = {}
        valid = True

        for src_lead, dst_lead in lead_map.items():
            if src_lead not in signal_dict:
                valid = False
                break
            sig = signal_dict[src_lead]
            if sig.shape[0] != 5000 or np.any(np.isnan(sig)):
                valid = False
                break
            current_sample[dst_lead] = sig.astype(np.float32)

        if valid:
            for dst_lead, signal in current_sample.items():
                lead_data[dst_lead].append(signal)
            labels.append(label)
        else:
            skipped += 1


# Save results
print(f"\n✅ Valid samples: {len(labels)}")
print(f"❌ Skipped samples: {skipped}")

expected_len = len(labels)
for lead_name, signals in lead_data.items():
    if len(signals) != expected_len:
        raise ValueError(f"❌ Mismatch in {lead_name}: {len(signals)} vs labels {expected_len}")
    tensor = torch.tensor(np.stack(signals), dtype=torch.float32)
    torch.save(tensor, os.path.join(output_dir, f"{lead_name}.pt"))
    print(f"✅ Saved {lead_name}.pt | Shape: {tensor.shape}")

labels_tensor = torch.tensor(labels, dtype=torch.long)
torch.save(labels_tensor, os.path.join(output_dir, "labels.pt"))
print("✅ Saved labels.pt")
