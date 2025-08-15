import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
import random
import numpy as np
import torch.nn.functional as nnf
import torch.optim as optim
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

class MMVAEPlus(nn.Module):
    def __init__(self, prior_dist, latent_dim, num_leads=12, input_dim_per_lead=5000):
        super(MMVAEPlus, self).__init__()
        self.encoders = nn.ModuleList([ECGLeadEncoder(input_dim=input_dim_per_lead, latent_dim=latent_dim) for _ in range(num_leads)])
        self.decoders = nn.ModuleList([ECGLeadDecoder(latent_dim=latent_dim, output_dim=input_dim_per_lead) for _ in range(num_leads)])
        self.pz = prior_dist
        self.latent_dim = latent_dim
        self.num_leads = num_leads

    def forward(self, inputs, modality_mask=None):
        qz_x_list, mus, logvars = [], [], []

        # Step 1: Encode each lead
        for i, (encoder, x) in enumerate(zip(self.encoders, inputs)):
            mu, logvar = encoder(x)
            qz_x_list.append((mu, logvar))
            mus.append(mu)
            logvars.append(logvar)

        mus = torch.stack(mus)      # Shape: (M, B, D)
        logvars = torch.stack(logvars)  # Shape: (M, B, D)

        # Step 2: Apply modality mask (drop modalities if needed)
        if modality_mask is not None:
            modality_mask = modality_mask.view(self.num_leads, 1, 1).to(mus.device)
            mus = mus * modality_mask
            logvars = logvars * modality_mask
            mask_sum = modality_mask.sum(dim=0)
            mask_sum[mask_sum == 0] = 1  # Avoid division by zero
        else:
            mask_sum = mus.new_tensor(self.num_leads).float()

        # Step 3: Joint posterior by averaging (Eq. 6 in the paper)
        mu_joint = mus.sum(dim=0) / mask_sum
        logvar_joint = logvars.sum(dim=0) / mask_sum  # This is not exact, but aligned with the simplified MMVAE+ implementation.

        # Step 4: Sample latent
        z_sample = self.sample_latent(mu_joint, logvar_joint)

        # Step 5: Decode into each lead
        recon_leads = [decoder(z_sample) for decoder in self.decoders]

        return qz_x_list, recon_leads, [z_sample]

    def sample_latent(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std



# Set seed for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Enable AMP (Automatic Mixed Precision)
scaler = GradScaler()


# Silhouette Score computation
def compute_silhouette_score_kmeans(latent_z, n_clusters=13, use_pca=True):
    latent_z_np = latent_z.detach().cpu().numpy()
    latent_z_np = StandardScaler().fit_transform(latent_z_np)
    if use_pca:
        latent_z_np = PCA(n_components=2).fit_transform(latent_z_np)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(latent_z_np)
    return silhouette_score(latent_z_np, kmeans.labels_)

# ELBO loss for ECG leads
def elbo_loss(recon_leads, lead_inputs, mu, logvar, lambda_ecg=1.0, annealing_factor=1.0):
    ecg_mse = 0
    eps = 1e-8
    for recon, lead in zip(recon_leads, lead_inputs):
        ecg_mse += nnf.mse_loss(recon, lead, reduction='sum')

    logvar = torch.clamp(logvar, min=-10, max=10)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp() + eps, dim=1)
    ELBO = torch.mean(lambda_ecg * ecg_mse / lead_inputs[0].size(0) + annealing_factor * KLD)
    return ELBO

# Loss tracking class
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# --- Train (no silhouette here) ---
def train(epoch, model, dataloader, optimizer, annealing_epochs,
          accumulation_steps=2, log_interval=10, lambda_recon=10.0):
    model.train()
    train_loss_meter = AverageMeter()
    N_mini_batches = len(dataloader)

    # Zero once at start so accumulation divides correctly across steps
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader):
        annealing_factor = min((epoch + 1) / (2 * annealing_epochs), 1.0)
        lead_inputs = [tensor.to(device, non_blocking=True) for tensor in batch.values()]

        with autocast():
            qz_x_list, px_z_list, latent_z_list = model(lead_inputs)

            # OPTION 2: use one modality's stats as placeholder for KL
            mu_joint, logvar_joint = qz_x_list[0]  # (B, D) each

            # Sanity check shapes
            for recon, lead in zip(px_z_list, lead_inputs):
                assert recon.shape == lead.shape, f"Shape mismatch: {recon.shape} vs {lead.shape}"

            loss = elbo_loss(px_z_list, lead_inputs, mu_joint, logvar_joint,
                             lambda_ecg=lambda_recon, annealing_factor=annealing_factor)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        # Step optimizer every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        train_loss_meter.update(loss.item(), len(lead_inputs[0]))

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(lead_inputs[0])}/{len(dataloader.dataset)} "
                  f"({100. * batch_idx / N_mini_batches:.0f}%)]\tLoss: {train_loss_meter.avg:.6f}")

    return train_loss_meter.avg

# ----------- MAIN TRAINING SETUP ---------------

# --- Hyperparams ---
n_latents = 256
epochs = 100
annealing_epochs = 50
lr = 5e-4
log_interval = 10
accumulation_steps = 2
lambda_recon = 10.0

# --- Model & Optimizer ---
prior_dist = prior_expert((n_latents,), use_cuda=torch.cuda.is_available())
model = MMVAEPlus(prior_dist, latent_dim=n_latents, num_leads=12, input_dim_per_lead=5000)
model = model.to(device)

# (Optional) torch.compile for PyTorch 2.x
try:
    model = torch.compile(model)
    print("Using JIT-compiled model.")
except Exception as e:
    print(f"torch.compile() failed: {e}\nProceeding without JIT.")

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)


# --- Train Loop ---
best_loss = float('inf')
save_dir = "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/pretrain"

for epoch in range(1, epochs + 1):
    train_loss = train(epoch, model, dataloader, optimizer, annealing_epochs,
                       accumulation_steps, log_interval, lambda_recon)
    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")

    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), f"{save_dir}/best_MMVAEPlus.pth")
        print(f"Saved best model with Loss: {best_loss:.6f}")

torch.save(model.state_dict(), f"{save_dir}/MMVAEPlus.pth")
print("Training complete. Model saved.")

