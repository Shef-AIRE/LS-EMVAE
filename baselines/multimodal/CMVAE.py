import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset


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
    "LEAD_V6": "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/12_lead_ecg/LEAD_V6.pt",
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


def prior_expert(size, use_cuda=False):
    if isinstance(size, int):
        size = (size,)
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


class ECGLeadEncoder(nn.Module):
    def __init__(self, input_dim=5000, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        conv_output_dim = input_dim // (2 ** 3)
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
        super().__init__()
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


class CMVAE(nn.Module):
    def __init__(
        self,
        prior_dist,
        latent_dim,
        num_leads=12,
        input_dim_per_lead=5000,
        latent_dim_w=64,
        num_clusters=20,
        beta=2.5,
    ):
        super().__init__()
        if latent_dim_w >= latent_dim:
            latent_dim_w = max(1, latent_dim // 2)
        self.latent_dim = latent_dim
        self.latent_dim_w = latent_dim_w
        self.latent_dim_z = latent_dim - latent_dim_w
        self.num_leads = num_leads
        self.num_clusters = num_clusters
        self.beta = beta
        self.pz = prior_dist
        self.encoders = nn.ModuleList(
            [ECGLeadEncoder(input_dim=input_dim_per_lead, latent_dim=latent_dim) for _ in range(num_leads)]
        )
        self.decoders = nn.ModuleList(
            [ECGLeadDecoder(latent_dim=latent_dim, output_dim=input_dim_per_lead) for _ in range(num_leads)]
        )
        self.cluster_logits = nn.Parameter(torch.zeros(num_clusters))
        self.cluster_mu = nn.Parameter(torch.randn(num_clusters, self.latent_dim_z) * 0.05)
        self.cluster_logvar_unconstrained = nn.Parameter(torch.zeros(num_clusters, self.latent_dim_z))
        self.aux_w_logvar_unconstrained = nn.Parameter(torch.zeros(num_leads, self.latent_dim_w))

    def sample_latent(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def split_latent(self, tensor):
        return torch.split(tensor, [self.latent_dim_w, self.latent_dim_z], dim=-1)

    def get_cluster_params(self):
        pi = torch.softmax(self.cluster_logits, dim=0)
        var = nnf.softplus(self.cluster_logvar_unconstrained) + 1e-6
        logvar = torch.log(var)
        return pi, self.cluster_mu, var, logvar

    def cross_decode(self, w_samples, z_samples):
        recon_matrix = []
        for src_idx in range(self.num_leads):
            src_row = []
            for dst_idx, decoder in enumerate(self.decoders):
                if src_idx == dst_idx:
                    w_decode = w_samples[src_idx]
                else:
                    w_var = nnf.softplus(self.aux_w_logvar_unconstrained[dst_idx]).unsqueeze(0) + 1e-6
                    w_std = torch.sqrt(w_var)
                    w_decode = torch.randn_like(w_samples[src_idx]) * w_std
                latent_for_decode = torch.cat([w_decode, z_samples[src_idx]], dim=-1)
                src_row.append(decoder(latent_for_decode))
            recon_matrix.append(src_row)
        return recon_matrix

    def forward(self, inputs):
        qz_x_list = []
        mu_w_list, logvar_w_list = [], []
        mu_z_list, logvar_z_list = [], []
        w_samples, z_samples = [], []
        for encoder, input_signal in zip(self.encoders, inputs):
            mu, logvar = encoder(input_signal)
            sample = self.sample_latent(mu, logvar)
            mu_w, mu_z = self.split_latent(mu)
            logvar_w, logvar_z = self.split_latent(logvar)
            w, z = self.split_latent(sample)
            qz_x_list.append((mu, logvar))
            mu_w_list.append(mu_w)
            logvar_w_list.append(logvar_w)
            mu_z_list.append(mu_z)
            logvar_z_list.append(logvar_z)
            w_samples.append(w)
            z_samples.append(z)
        recon_matrix = self.cross_decode(w_samples, z_samples)
        latent_stats = {
            "mu_w": torch.stack(mu_w_list),
            "logvar_w": torch.stack(logvar_w_list),
            "mu_z": torch.stack(mu_z_list),
            "logvar_z": torch.stack(logvar_z_list),
            "w_samples": torch.stack(w_samples),
            "z_samples": torch.stack(z_samples),
        }
        return qz_x_list, recon_matrix, latent_stats


def gaussian_log_prob_diag(x, mu, logvar):
    return -0.5 * (math.log(2.0 * math.pi) + logvar + ((x - mu) ** 2) / torch.exp(logvar)).sum(dim=-1)


def cmvae_loss(model, recon_matrix, lead_inputs, latent_stats, lambda_ecg=1.0, annealing_factor=1.0):
    num_modalities = len(lead_inputs)
    batch_size = lead_inputs[0].size(0)
    recon_loss = 0.0
    for src_idx in range(num_modalities):
        for dst_idx in range(num_modalities):
            recon_loss = recon_loss + nnf.mse_loss(recon_matrix[src_idx][dst_idx], lead_inputs[dst_idx], reduction="sum")
    recon_loss = recon_loss / (batch_size * num_modalities)

    mu_w = latent_stats["mu_w"]
    logvar_w = torch.clamp(latent_stats["logvar_w"], min=-10.0, max=10.0)
    kl_w = -0.5 * torch.sum(1.0 + logvar_w - mu_w.pow(2) - logvar_w.exp(), dim=-1).mean()

    mu_z = latent_stats["mu_z"]
    logvar_z = torch.clamp(latent_stats["logvar_z"], min=-10.0, max=10.0)
    z_samples = latent_stats["z_samples"]
    pi, cluster_mu, cluster_var, cluster_logvar = model.get_cluster_params()
    log_pi = torch.log(pi + 1e-8)

    kl_z = 0.0
    for m in range(mu_z.size(0)):
        mu_m = mu_z[m]
        logvar_m = logvar_z[m]
        var_m = torch.exp(logvar_m).unsqueeze(1)
        mu_m_expanded = mu_m.unsqueeze(1)
        cluster_mu_expanded = cluster_mu.unsqueeze(0)
        cluster_var_expanded = cluster_var.unsqueeze(0)
        cluster_logvar_expanded = cluster_logvar.unsqueeze(0)
        diff_sq = (mu_m_expanded - cluster_mu_expanded).pow(2)
        kl_q_c = 0.5 * torch.sum(
            cluster_logvar_expanded - logvar_m.unsqueeze(1) + (var_m + diff_sq) / cluster_var_expanded - 1.0,
            dim=-1,
        )
        log_p_zc = log_pi.unsqueeze(0) + gaussian_log_prob_diag(
            z_samples[m].unsqueeze(1), cluster_mu_expanded, cluster_logvar_expanded
        )
        gamma = torch.softmax(log_p_zc, dim=-1)
        kl_mix = torch.sum(gamma * (kl_q_c + torch.log(gamma + 1e-8) - log_pi.unsqueeze(0)), dim=-1)
        kl_z = kl_z + kl_mix.mean()
    kl_z = kl_z / mu_z.size(0)

    total_kl = kl_w + kl_z
    total = lambda_ecg * recon_loss + annealing_factor * model.beta * total_kl
    return total, recon_loss.detach(), kl_w.detach(), kl_z.detach()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
scaler = GradScaler()


def train(epoch, model, dataloader, optimizer, annealing_epochs, accumulation_steps=2, log_interval=10):
    model.train()
    total_meter = AverageMeter()
    recon_meter = AverageMeter()
    klw_meter = AverageMeter()
    klz_meter = AverageMeter()
    n_mini_batches = len(dataloader)

    optimizer.zero_grad()
    for batch_idx, batch in enumerate(dataloader):
        annealing_factor = min((epoch + 1) / (2 * annealing_epochs), 1.0)
        lead_inputs = [tensor.to(device) for tensor in batch.values()]

        with autocast():
            _, recon_matrix, latent_stats = model(lead_inputs)
            loss, recon_loss, kl_w, kl_z = cmvae_loss(
                model,
                recon_matrix,
                lead_inputs,
                latent_stats,
                lambda_ecg=10.0,
                annealing_factor=annealing_factor,
            )
            scaled_loss = loss / accumulation_steps

        scaler.scale(scaled_loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_n = len(lead_inputs[0])
        total_meter.update(loss.item(), batch_n)
        recon_meter.update(recon_loss.item(), batch_n)
        klw_meter.update(kl_w.item(), batch_n)
        klz_meter.update(kl_z.item(), batch_n)

        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * batch_n}/{len(dataloader.dataset)} "
                f"({100.0 * batch_idx / n_mini_batches:.0f}%)] "
                f"Loss: {total_meter.avg:.6f} Recon: {recon_meter.avg:.6f} "
                f"KL_w: {klw_meter.avg:.6f} KL_z: {klz_meter.avg:.6f}"
            )

    return total_meter.avg


n_latents = 256
epochs = 100
annealing_epochs = 50
lr = 5e-4
log_interval = 10
accumulation_steps = 2

params = {"latent_dim": n_latents}
prior_dist = prior_expert((n_latents,), use_cuda=torch.cuda.is_available())
model = CMVAE(
    prior_dist=prior_dist,
    latent_dim=n_latents,
    num_leads=12,
    input_dim_per_lead=5000,
    latent_dim_w=64,
    num_clusters=20,
    beta=2.5,
)
model = model.to(device, memory_format=torch.channels_last)

try:
    model = torch.compile(model)
    print("Using JIT-compiled model.")
except Exception as e:
    print(f"torch.compile() failed: {e}")
    print("Proceeding without JIT compilation.")

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

best_loss = float("inf")
for epoch in range(1, epochs + 1):
    train_loss = train(epoch, model, dataloader, optimizer, annealing_epochs, accumulation_steps, log_interval)
    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(
            model.state_dict(),
            "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/pretrain/best_CMVAE.pth",
        )
        print(f"Saved best model with Loss: {best_loss:.6f}")

torch.save(
    model.state_dict(),
    "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/pretrain/CMVAE.pth",
)
print("Training complete. Model saved.")
