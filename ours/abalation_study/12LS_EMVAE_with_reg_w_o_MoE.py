import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import random
import numpy as np
import torch.nn.functional as nnf
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# --- Data Loading ---
lead_file_paths = {
    "LEAD_I": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_I.pt",
    "LEAD_II": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_II.pt",
    "LEAD_III": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_III.pt",
    "LEAD_aVR": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_aVR.pt",
    "LEAD_aVL": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_aVL.pt",
    "LEAD_aVF": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_aVF.pt",
    "LEAD_V1": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_V1.pt",
    "LEAD_V2": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_V2.pt",
    "LEAD_V3": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_V3.pt",
    "LEAD_V4": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_V4.pt",
    "LEAD_V5": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_V5.pt",
    "LEAD_V6": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_V6.pt"
}

ecg_lead_tensors = {lead: torch.load(path) for lead, path in lead_file_paths.items()}
sample_count = len(next(iter(ecg_lead_tensors.values())))
for tensor in ecg_lead_tensors.values():
    assert len(tensor) == sample_count, "All leads must have the same number of samples."

class ECGMultiLeadDataset(Dataset):
    def __init__(self, ecg_leads):
        self.ecg_leads = ecg_leads

    def __len__(self):
        return len(next(iter(self.ecg_leads.values())))

    def __getitem__(self, idx):
        return {lead: self.ecg_leads[lead][idx].unsqueeze(0) for lead in self.ecg_leads}

dataset = ECGMultiLeadDataset(ecg_lead_tensors)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

print("Data loading done")


# --- PoE (Product of Experts) ---
class ProductOfExperts(nn.Module):
    def forward(self, mus, logvars, eps=1e-8):
        var = torch.exp(logvars) + eps
        T = 1. / var
        mu_poe = torch.sum(mus * T, dim=0) / torch.sum(T, dim=0)
        var_poe = 1. / torch.sum(T, dim=0)
        logvar_poe = torch.log(var_poe + eps)
        return mu_poe, logvar_poe

def prior_expert(size, use_cuda=False):
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar

class ECGLeadEncoder(nn.Module):
    def __init__(self, input_dim=5000, latent_dim=256):
        super(ECGLeadEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        conv_output_dim = 5000 // (2 ** 3)  
        self.fc_mu = nn.Linear(64 * conv_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(64 * conv_output_dim, latent_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class SharedDecoder(nn.Module):
    def __init__(self, latent_dim=256, output_dim=5000):
        super(SharedDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64 * (output_dim // 8))
        self.convtrans1 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.convtrans2 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.convtrans3 = nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.output_activation = nn.Identity()
    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, z.size(1) // 64)
        z = self.relu(self.convtrans1(z))
        z = self.relu(self.convtrans2(z))
        z = self.output_activation(self.convtrans3(z))
        return z

import torch
import torch.nn as nn
from torch.autograd import Variable

class LSEMVAE_POE(nn.Module):
    def __init__(self, input_dim_per_lead=5000, latent_dim=128, num_leads=12):
        super(LSEMVAE_POE, self).__init__()
        self.num_leads = num_leads
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleList([
            ECGLeadEncoder(input_dim=input_dim_per_lead, latent_dim=latent_dim)
            for _ in range(num_leads)
        ])
        self.decoder = SharedDecoder(latent_dim=latent_dim, output_dim=input_dim_per_lead)
        self.experts = ProductOfExperts()

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, inputs):  # inputs: list of ECG signals, each of shape (B, input_dim_per_lead)
        mu, logvar = self.infer(inputs)
        z = self.reparametrize(mu, logvar)
        recon_signals = [self.decoder(z) for _ in range(self.num_leads)]
        return recon_signals, mu, logvar

    def infer(self, inputs):  # inputs: list of tensors
        batch_size = inputs[0].size(0)
        use_cuda = next(self.parameters()).is_cuda

        mu_stack, logvar_stack = prior_expert((1, batch_size, self.latent_dim), use_cuda)

        for i, input_signal in enumerate(inputs):
            lead_mu, lead_logvar = self.encoders[i](input_signal)
            mu_stack = torch.cat((mu_stack, lead_mu.unsqueeze(0)), dim=0)
            logvar_stack = torch.cat((logvar_stack, lead_logvar.unsqueeze(0)), dim=0)

        mu_combined, logvar_combined = self.experts(mu_stack, logvar_stack)
        return mu_combined, logvar_combined


class ProductOfExperts(nn.Module):
    def forward(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


def prior_expert(size, use_cuda=False):
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


print("model loading done")


# Set seed for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Prior Expert function
def prior_expert(size, use_cuda=False):
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar

# ELBO loss function
def elbo_loss(recon_leads, lead_inputs, mu, logvar, batch_size, lambda_recon=1.0, annealing_factor=1.0, scale_factor=1e-3):
    mse_loss = sum(nnf.mse_loss(recon, lead, reduction='sum') for recon, lead in zip(recon_leads, lead_inputs))
    mse_loss = (lambda_recon * mse_loss * scale_factor) / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_loss = kl_loss.sum() / batch_size
    return mse_loss + annealing_factor * kl_loss

# Utility to track training loss
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Training loop
def train(epoch, model, dataloader, optimizer, annealing_epochs, accumulation_steps=2, log_interval=10):
    model.train()
    train_loss_meter = AverageMeter()
    N_mini_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        annealing_factor = min(epoch / annealing_epochs, 1.0)
        lead_inputs = [tensor.to(device) for tensor in batch.values()]
        optimizer.zero_grad()

        with autocast():
            recon_leads, mu, logvar = model(lead_inputs)
            batch_size = lead_inputs[0].size(0)
            total_loss = elbo_loss(recon_leads, lead_inputs, mu, logvar, batch_size, annealing_factor=annealing_factor)
            total_loss = total_loss / accumulation_steps

        scaler.scale(total_loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss_meter.update(total_loss.item(), batch_size)

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * batch_size}/{len(dataloader.dataset)} "
                  f"({100. * batch_idx / N_mini_batches:.0f}%)]\tLoss: {train_loss_meter.avg:.6f}")

    return train_loss_meter.avg


n_latents = 256
epochs = 100
annealing_epochs = 50
lr = 5e-4
log_interval = 10
batch_size = 1024
accumulation_steps = 2

model = LSEMVAE_POE(latent_dim=n_latents, num_leads=12, input_dim_per_lead=5000)
model = model.to(device, memory_format=torch.channels_last)

try:
    model = torch.compile(model)
    print("Using JIT-compiled model.")
except Exception as e:
    print(f"torch.compile() failed: {e}")
    print("Proceeding without JIT compilation.")

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
scaler = GradScaler()

# === Training ===
best_loss = float('inf')

for epoch in range(1, epochs + 1):
    train_loss = train(epoch, model, dataloader, optimizer, annealing_epochs, accumulation_steps, log_interval)
    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), '/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/pretrain/best_LS_EMVAE_with_reg_12_w_o_moe.pth')
        print(f"Saved best model with Loss: {best_loss:.6f}")

torch.save(model.state_dict(), '/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/pretrain/LS_EMVAE_with_reg_12_w_o_moe.pth')
print("Training complete. Model saved.")

