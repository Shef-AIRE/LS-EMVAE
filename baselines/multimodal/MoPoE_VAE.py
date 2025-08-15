import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
import torch.nn.functional as nnf
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast



# Define the file paths for the 12-lead ECG data
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

# Load all lead tensors
ecg_lead_tensors = {lead: torch.load(path) for lead, path in lead_file_paths.items()}

# Ensure all leads have the same number of samples
sample_count = len(next(iter(ecg_lead_tensors.values())))
for tensor in ecg_lead_tensors.values():
    assert len(tensor) == sample_count, "All leads must have the same number of samples."

class ECGMultiLeadDataset(Dataset):
    def __init__(self, ecg_leads):
        self.ecg_leads = ecg_leads

    def __len__(self):
        return len(next(iter(self.ecg_leads.values())))

    def __getitem__(self, idx):
        # Return each lead sample with the correct input shape (batch_size, 1, 5000)
        return {lead: self.ecg_leads[lead][idx].unsqueeze(0) for lead in self.ecg_leads}

# Initialize the dataset and dataloader
dataset = ECGMultiLeadDataset(ecg_lead_tensors)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

class ProductOfExperts(nn.Module):
    def forward(self, mus, logvars, eps=1e-8):
        """Apply Product of Experts (PoE) for a subset of modalities."""
        var = torch.exp(logvars) + eps
        T = 1. / var
        mu_poe = torch.sum(mus * T, dim=0) / torch.sum(T, dim=0)
        var_poe = 1. / torch.sum(T, dim=0)
        logvar_poe = torch.log(var_poe + eps)
        return mu_poe, logvar_poe

def mixture_of_experts(mus, logvars, weights=None):
    """Mixture of Experts (MoE) applied after PoE."""
    num_experts = mus.shape[0]

    if weights is None:
        weights = torch.ones(num_experts).to(mus.device) / num_experts

    weights = weights / torch.sum(weights)
    weights = weights.view(num_experts, 1, 1)

    combined_mu = torch.sum(weights * mus, dim=0)
    combined_logvar = torch.sum(weights * logvars, dim=0)

    return combined_mu, combined_logvar

def prior_expert(size, use_cuda=False):
    """Define the prior expert (standard normal)."""
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

class ECGLeadDecoder(nn.Module):
    def __init__(self, latent_dim=256, output_dim=5000):
        super(ECGLeadDecoder, self).__init__()
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

class MoPoE(nn.Module):
    def __init__(self, prior_dist, latent_dim, num_leads=12, input_dim_per_lead=5000):
        super(MoPoE, self).__init__()
        self.encoders = nn.ModuleList([ECGLeadEncoder(input_dim=input_dim_per_lead, latent_dim=latent_dim) for _ in range(num_leads)])
        self.decoders = nn.ModuleList([ECGLeadDecoder(latent_dim=latent_dim, output_dim=input_dim_per_lead) for _ in range(num_leads)])
        self.pz = prior_dist
        self.poe = ProductOfExperts()
        self.latent_dim = latent_dim

    def forward(self, inputs):
        qz_x_list, latent_z_list = [], []
        mus, logvars = [], []

        for encoder, input_signal in zip(self.encoders, inputs):
            mu, logvar = encoder(input_signal)
            qz_x_list.append((mu, logvar))
            mus.append(mu)
            logvars.append(logvar)

        prior_mu, prior_logvar = prior_expert(mus[0].shape, use_cuda=torch.cuda.is_available())
        prior_mu = prior_mu.to(mus[0].device)
        prior_logvar = prior_logvar.to(logvars[0].device)
        mus.append(prior_mu)
        logvars.append(prior_logvar)

        mus = torch.stack(mus)  # Shape: (M, B, D)
        logvars = torch.stack(logvars)  # Shape: (M, B, D)

        # Step 1: Apply Product of Experts (PoE) in modality groups
        num_experts = mus.shape[0]
        num_groups = min(4, num_experts)  # Split into 4 groups max
        group_size = num_experts // num_groups

        poe_mus, poe_logvars = [], []
        for i in range(num_groups):
            subset_mus = mus[i * group_size:(i + 1) * group_size]
            subset_logvars = logvars[i * group_size:(i + 1) * group_size]
            mu_poe, logvar_poe = self.poe(subset_mus, subset_logvars)
            poe_mus.append(mu_poe)
            poe_logvars.append(logvar_poe)

        # Stack PoE outputs
        poe_mus = torch.stack(poe_mus)  # Shape: (num_groups, B, D)
        poe_logvars = torch.stack(poe_logvars)  # Shape: (num_groups, B, D)

        # Step 2: Apply Mixture of Experts (MoE) over PoE results
        mu_moe, logvar_moe = mixture_of_experts(poe_mus, poe_logvars)

        # Sample latent representation
        z_sample = self.sample_latent(mu_moe, logvar_moe)

        # Decode into all leads using separate decoders
        recon_leads = [decoder(z_sample) for decoder in self.decoders]

        return qz_x_list, recon_leads, [z_sample]

    def sample_latent(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
print("done loading data and model")


# Set the seed for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = False  # Allow faster convolutions
    torch.backends.cudnn.benchmark = True  # Speed up training

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable AMP (Automatic Mixed Precision)
scaler = GradScaler()

# Optimized ELBO loss function
def elbo_loss(recon_leads, lead_inputs, mu, logvar, lambda_ecg=1.0, annealing_factor=1.0):
    ecg_mse = sum(nnf.mse_loss(recon, lead, reduction='sum') for recon, lead in zip(recon_leads, lead_inputs))
    logvar = torch.clamp(logvar, min=-10, max=10)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return torch.mean(lambda_ecg * ecg_mse / lead_inputs[0].size(0) + annealing_factor * KLD)

# Utility class to track average loss
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val, self.sum, self.count = val, self.sum + val * n, self.count + n
        self.avg = self.sum / self.count

# Optimized training function (No Silhouette Score)
def train(epoch, model, dataloader, optimizer, annealing_epochs, accumulation_steps=2, log_interval=10):
    model.train()
    train_loss_meter = AverageMeter()
    N_mini_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        annealing_factor = min((epoch + 1) / (2 * annealing_epochs), 1.0)
        lead_inputs = [tensor.to(device) for tensor in batch.values()]

        optimizer.zero_grad()

        with autocast():
            qz_x_list, px_z_list, latent_z_list = model(lead_inputs)

            mus = torch.stack([qz[0] for qz in qz_x_list])
            logvars = torch.stack([qz[1] for qz in qz_x_list])

            mu_joint = torch.mean(mus, dim=0)
            logvar_joint = torch.mean(logvars, dim=0)

            loss = elbo_loss(px_z_list, lead_inputs, mu_joint, logvar_joint, lambda_ecg=10.0, annealing_factor=annealing_factor)
            loss = loss / accumulation_steps  # Normalize for accumulation

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss_meter.update(loss.item(), len(lead_inputs[0]))

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(lead_inputs[0])}/{len(dataloader.dataset)} "
                  f"({100. * batch_idx / N_mini_batches:.0f}%)]\tLoss: {train_loss_meter.avg:.6f}")

    return train_loss_meter.avg

# Hyperparameters (Optimized)
n_latents = 256
epochs = 100
annealing_epochs = 50
lr = 5e-4  # Increased learning rate for faster convergence
log_interval = 10
batch_size = 1024  # Increased for speed
accumulation_steps = 2  # Reduce memory footprint

# Model and optimizer setup
params = {'latent_dim': n_latents}
prior_dist = prior_expert((n_latents,), use_cuda=torch.cuda.is_available())
model = MoPoE(prior_dist, latent_dim=n_latents, num_leads=12, input_dim_per_lead=5000)
model = model.to(device, memory_format=torch.channels_last)

# Try enabling JIT Compilation (torch.compile()), fallback if not supported
try:
    model = torch.compile(model)  # JIT optimization for PyTorch 2.0+
    print("Using JIT-compiled model.")
except Exception as e:
    print(f"torch.compile() failed: {e}")
    print("Proceeding without JIT compilation.")

# Use AdamW optimizer for better performance with large batches
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

# Training loop (No Silhouette Score)
best_loss = float('inf')

for epoch in range(1, epochs + 1):
    train_loss = train(epoch, model, dataloader, optimizer, annealing_epochs, accumulation_steps, log_interval)
    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")

    # Save best model based on lowest loss
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), '/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/pretrain/best_MoPoE.pth')
        print(f"Saved best model with Loss: {best_loss:.6f}")

# Save final model after training
torch.save(model.state_dict(), '/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/pretrain/MoPoE.pth')
print("Training complete. Model saved.")


