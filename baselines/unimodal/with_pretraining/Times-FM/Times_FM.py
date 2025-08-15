import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler
import gc
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from huggingface_hub import snapshot_download
from torch.cuda.amp import autocast, GradScaler

from src.timesfm.timesfm_base import TimesFmCheckpoint, TimesFmHparams
from src.timesfm.timesfm_torch import TimesFmTorch
from src.timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# --------------------
# Load Your Dataset
# --------------------

#Change dataset path for other downstream tasks

lead_file_paths = {
    "LEAD_I": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/LEAD_I.pt",
    "LEAD_II": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/LEAD_II.pt",
    "LEAD_III": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/LEAD_III.pt",
    "LEAD_aVR": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/LEAD_aVR.pt",
    "LEAD_aVL": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/LEAD_aVL.pt",
    "LEAD_aVF": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/LEAD_aVF.pt",
    "LEAD_V1": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/LEAD_V1.pt",
    "LEAD_V2": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/LEAD_V2.pt",
    "LEAD_V3": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/LEAD_V3.pt",
    "LEAD_V4": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/LEAD_V4.pt",
    "LEAD_V5": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/LEAD_V5.pt",
    "LEAD_V6": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/LEAD_V6.pt"
}
labels_file_path = "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/finetune/data_aspire_PAP/labels.pt"

ecg_lead_tensors = {lead: torch.load(path) for lead, path in lead_file_paths.items()}
labels = torch.load(labels_file_path)
sample_count = len(labels)
for lead_tensor in ecg_lead_tensors.values():
    assert lead_tensor.shape[0] == sample_count

class ECGMultiLeadDatasetWithLabels(torch.utils.data.Dataset):
    def __init__(self, ecg_leads, labels):
        self.ecg_leads = ecg_leads
        self.labels = labels
        self.lead_names = list(ecg_leads.keys())

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        signal = torch.cat([self.ecg_leads[lead][idx].unsqueeze(0) for lead in self.lead_names], dim=0)
        input_padding = torch.zeros_like(signal)
        freq = torch.tensor([1], dtype=torch.long)
        return signal, input_padding, freq, self.labels[idx]

def get_model(rank):
    repo_id = "google/timesfm-2.0-500m-pytorch"
    hparams = TimesFmHparams(
        backend="cuda",
        per_core_batch_size=32,
        context_len=192,
        horizon_len=1,
        use_positional_embedding=False,
        num_layers=50,
    )
    tfm = TimesFmTorch(hparams=hparams, checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id))
    model = PatchedTimeSeriesDecoder(tfm._model_config)
    ckpt_path = os.path.join(snapshot_download(repo_id), "torch_model.ckpt")
    ckpt = torch.load(ckpt_path, map_location=f"cuda:{rank}")
    model.load_state_dict(ckpt, strict=False)
    return model.to(rank)

def setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    dataset = ECGMultiLeadDatasetWithLabels(ecg_lead_tensors, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_data, batch_size=32, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, sampler=val_sampler, num_workers=4)

    model = get_model(rank)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    for epoch in range(10):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0

        for x, pad, freq, y in train_loader:
            x, pad, freq, y = x.to(device), pad.to(device), freq.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(x, pad, freq)[..., 0][:, -1, :]
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            del x, pad, freq, y, logits, loss
            torch.cuda.empty_cache()
            gc.collect()

        if rank == 0:
            print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation (only on rank 0)
    if rank == 0:
        model.eval()
        y_true, y_pred, y_score = [], [], []
        with torch.no_grad(), autocast():
            for x, pad, freq, y in val_loader:
                x, pad, freq = x.to(device), pad.to(device), freq.to(device)
                logits = model(x, pad, freq)[..., 0][:, -1, :]
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                y_true.extend(y.numpy())
                y_pred.extend(preds.cpu().numpy())
                y_score.extend(probs[:, 1].cpu().numpy())

        print(f"\nValidation Accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"Validation MCC: {matthews_corrcoef(y_true, y_pred):.4f}")
        try:
            print(f"Validation AUROC: {roc_auc_score(y_true, y_score):.4f}")
        except ValueError:
            print("Validation AUROC: N/A")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)
