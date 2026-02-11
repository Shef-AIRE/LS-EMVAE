import copy
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as nnf
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast


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


def prior_expert(size, use_cuda=False):
    if isinstance(size, int):
        size = (size,)
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
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


class LeadViTEncoder(nn.Module):
    def __init__(self, input_dim=5000, patch_size=20, embed_dim=256, depth=4, num_heads=8):
        super(LeadViTEncoder, self).__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.num_patches = input_dim // patch_size
        self.patch_embed = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ),
            num_layers=depth
        )
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _fix_length(self, x):
        target = self.num_patches * self.patch_size
        if x.size(-1) > target:
            x = x[..., :target]
        elif x.size(-1) < target:
            x = nnf.pad(x, (0, target - x.size(-1)))
        return x

    def forward_tokens(self, x):
        x = self._fix_length(x)
        tokens = self.patch_embed(x).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0], tokens[:, 1:]

    def forward(self, x):
        cls, _ = self.forward_tokens(x)
        return cls, torch.zeros_like(cls)


class PatchDecoder(nn.Module):
    def __init__(self, embed_dim=256, patch_size=20):
        super(PatchDecoder, self).__init__()
        self.pre = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.recon_head = nn.Linear(embed_dim, patch_size)
        self.noise_head = nn.Linear(embed_dim, patch_size)

    def forward(self, tokens, sigma_embed):
        h = self.pre(tokens)
        if sigma_embed is not None:
            h = h + sigma_embed.unsqueeze(1)
        recon = self.recon_head(h)
        noise = self.noise_head(h)
        return recon, noise


class MCMAE(nn.Module):
    def __init__(
        self,
        prior_dist,
        latent_dim,
        num_leads=12,
        input_dim_per_lead=5000,
        patch_size=20,
        encoder_depth=4,
        num_heads=8,
        mask_ratio=0.8,
        tau=0.1,
        sigma_max=0.5
    ):
        super(MCMAE, self).__init__()
        self.pz = prior_dist
        self.latent_dim = latent_dim
        self.num_leads = num_leads
        self.input_dim_per_lead = input_dim_per_lead
        self.patch_size = patch_size
        self.num_patches = input_dim_per_lead // patch_size
        self.mask_ratio = mask_ratio
        self.tau = tau
        self.sigma_max = sigma_max
        self.encoders = nn.ModuleList(
            [LeadViTEncoder(input_dim=input_dim_per_lead, patch_size=patch_size, embed_dim=latent_dim, depth=encoder_depth, num_heads=num_heads) for _ in range(num_leads)]
        )
        self.projectors = nn.ModuleList(
            [nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)) for _ in range(num_leads)]
        )
        self.patch_decoders = nn.ModuleList([PatchDecoder(embed_dim=latent_dim, patch_size=patch_size) for _ in range(num_leads)])
        self.signal_decoders = nn.ModuleList([ECGLeadDecoder(latent_dim=latent_dim, output_dim=input_dim_per_lead) for _ in range(num_leads)])
        self.sigma_mlp = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim))
        self.teacher_encoders = None

    def patchify(self, x):
        target_len = self.num_patches * self.patch_size
        if x.size(-1) > target_len:
            x = x[..., :target_len]
        elif x.size(-1) < target_len:
            x = nnf.pad(x, (0, target_len - x.size(-1)))
        patches = x.unfold(-1, self.patch_size, self.patch_size)
        return patches.squeeze(1)

    def unpatchify(self, patches):
        signal = patches.reshape(patches.size(0), 1, self.num_patches * self.patch_size)
        if signal.size(-1) < self.input_dim_per_lead:
            signal = nnf.pad(signal, (0, self.input_dim_per_lead - signal.size(-1)))
        elif signal.size(-1) > self.input_dim_per_lead:
            signal = signal[..., :self.input_dim_per_lead]
        return signal

    def sample_mask(self, bsz, device):
        rand = torch.rand(bsz, self.num_patches, device=device)
        mask = rand < self.mask_ratio
        return mask

    def sigma_embedding(self, sigma):
        sigma = sigma.view(-1, 1)
        half = self.latent_dim // 2
        freq = torch.exp(torch.arange(half, device=sigma.device, dtype=sigma.dtype) * (-np.log(10000.0) / max(half - 1, 1)))
        args = sigma * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.size(1) < self.latent_dim:
            emb = nnf.pad(emb, (0, self.latent_dim - emb.size(1)))
        elif emb.size(1) > self.latent_dim:
            emb = emb[:, :self.latent_dim]
        return self.sigma_mlp(emb)

    def forward(self, inputs):
        qz_x_list = []
        lead_latents = []
        for enc, signal in zip(self.encoders, inputs):
            mu, _ = enc(signal)
            qz_x_list.append((mu, torch.zeros_like(mu)))
            lead_latents.append(mu)
        shared_latent = torch.stack(lead_latents, dim=0).mean(dim=0)
        recon_leads = [dec(shared_latent) for dec in self.signal_decoders]
        return qz_x_list, recon_leads, [shared_latent]

    def stage1_contrastive_loss(self, inputs, max_pairs=24):
        tokens_per_mod = []
        for m, signal in enumerate(inputs):
            _, tokens = self.encoders[m].forward_tokens(signal)
            proj = self.projectors[m](tokens)
            proj = nnf.normalize(proj, dim=-1)
            tokens_per_mod.append(proj.reshape(-1, self.latent_dim))

        all_pairs = list(itertools.combinations(range(self.num_leads), 2))
        if len(all_pairs) > max_pairs:
            all_pairs = random.sample(all_pairs, max_pairs)

        total = 0.0
        for i, j in all_pairs:
            zi = tokens_per_mod[i]
            zj = tokens_per_mod[j]
            logits_ij = zi @ zj.t() / self.tau
            logits_ji = logits_ij.t()
            labels = torch.arange(logits_ij.size(0), device=logits_ij.device)
            loss_ij = nnf.cross_entropy(logits_ij, labels)
            loss_ji = nnf.cross_entropy(logits_ji, labels)
            total = total + 0.5 * (loss_ij + loss_ji)

        return total / max(len(all_pairs), 1)

    def build_stage1_teacher(self):
        self.teacher_encoders = copy.deepcopy(self.encoders)
        for enc in self.teacher_encoders:
            enc.eval()
            for p in enc.parameters():
                p.requires_grad = False

    def stage2_losses(self, inputs, alpha=1.0, beta=1.0, gamma=1.0):
        patches_clean = []
        masks = []
        eps_targets = []
        sigmas = []
        cls_stage2 = []
        tokens_stage2 = []

        for signal in inputs:
            clean = self.patchify(signal)
            bsz = clean.size(0)
            mask = self.sample_mask(bsz, clean.device)
            sigma = torch.rand(bsz, 1, 1, device=clean.device) * self.sigma_max
            eps = torch.randn_like(clean)
            noisy = clean + sigma * eps
            noisy_signal = self.unpatchify(noisy)
            cls, tokens = self.encoders[len(tokens_stage2)].forward_tokens(noisy_signal)
            cls_stage2.append(cls)
            tokens_stage2.append(tokens)
            patches_clean.append(clean)
            masks.append(mask)
            eps_targets.append(sigma * eps)
            sigmas.append(sigma.squeeze(-1).squeeze(-1))

        tokens_stack = torch.stack(tokens_stage2, dim=0)
        fused_tokens = tokens_stack.mean(dim=0)

        recon_loss = 0.0
        denoise_loss = 0.0

        for m in range(self.num_leads):
            sigma_embed = self.sigma_embedding(sigmas[m])
            pred_recon, pred_noise = self.patch_decoders[m](fused_tokens, sigma_embed)
            target_clean = patches_clean[m]
            target_noise = eps_targets[m]
            mask = masks[m].unsqueeze(-1).float()
            inv_mask = 1.0 - mask
            recon_num = ((pred_recon - target_clean) ** 2 * mask).sum()
            recon_den = mask.sum() * self.patch_size + 1e-8
            denoise_num = ((pred_noise - target_noise) ** 2 * inv_mask).sum()
            denoise_den = inv_mask.sum() * self.patch_size + 1e-8
            recon_loss = recon_loss + recon_num / recon_den
            denoise_loss = denoise_loss + denoise_num / denoise_den

        recon_loss = recon_loss / self.num_leads
        denoise_loss = denoise_loss / self.num_leads

        distill_loss = torch.tensor(0.0, device=inputs[0].device)
        if self.teacher_encoders is not None:
            loss_terms = []
            with torch.no_grad():
                cls_stage1 = [self.teacher_encoders[m](inputs[m])[0] for m in range(self.num_leads)]
            for m in range(self.num_leads):
                loss_terms.append(nnf.smooth_l1_loss(cls_stage2[m], cls_stage1[m], beta=2.0))
            distill_loss = torch.stack(loss_terms).mean()

        total = alpha * recon_loss + beta * denoise_loss + gamma * distill_loss
        return total, recon_loss.detach(), denoise_loss.detach(), distill_loss.detach()


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
        self.avg = self.sum / max(self.count, 1)


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


def train_stage1(epoch, model, dataloader, optimizer, accumulation_steps=2, log_interval=10):
    model.train()
    loss_meter = AverageMeter()
    n_mini_batches = len(dataloader)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        lead_inputs = [tensor.to(device) for tensor in batch.values()]
        with autocast():
            loss = model.stage1_contrastive_loss(lead_inputs, max_pairs=24)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_meter.update(loss.item(), len(lead_inputs[0]))

        if batch_idx % log_interval == 0:
            print(f"Stage1 Epoch: {epoch} [{batch_idx * len(lead_inputs[0])}/{len(dataloader.dataset)} ({100. * batch_idx / n_mini_batches:.0f}%)]\tLoss: {loss_meter.avg:.6f}")

    return loss_meter.avg


def train_stage2(epoch, model, dataloader, optimizer, alpha, beta, gamma, accumulation_steps=2, log_interval=10):
    model.train()
    total_meter = AverageMeter()
    recon_meter = AverageMeter()
    denoise_meter = AverageMeter()
    distill_meter = AverageMeter()
    n_mini_batches = len(dataloader)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        lead_inputs = [tensor.to(device) for tensor in batch.values()]
        with autocast():
            total, l_recon, l_denoise, l_distill = model.stage2_losses(lead_inputs, alpha=alpha, beta=beta, gamma=gamma)
            loss = total / accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        bsz = len(lead_inputs[0])
        total_meter.update(total.item(), bsz)
        recon_meter.update(l_recon.item(), bsz)
        denoise_meter.update(l_denoise.item(), bsz)
        distill_meter.update(l_distill.item(), bsz)

        if batch_idx % log_interval == 0:
            print(
                f"Stage2 Epoch: {epoch} [{batch_idx * bsz}/{len(dataloader.dataset)} ({100. * batch_idx / n_mini_batches:.0f}%)]\t"
                f"Loss: {total_meter.avg:.6f} Recon: {recon_meter.avg:.6f} Denoise: {denoise_meter.avg:.6f} Distill: {distill_meter.avg:.6f}"
            )

    return total_meter.avg


n_latents = 256
stage1_epochs = 30
stage2_epochs = 70
lr = 5e-4
log_interval = 10
accumulation_steps = 2
alpha = 1.0
beta = 1.0
gamma = 1.0

prior_dist = prior_expert((n_latents,), use_cuda=torch.cuda.is_available())
model = MCMAE(
    prior_dist=prior_dist,
    latent_dim=n_latents,
    num_leads=12,
    input_dim_per_lead=5000,
    patch_size=20,
    encoder_depth=4,
    num_heads=8,
    mask_ratio=0.8,
    tau=0.1,
    sigma_max=0.5
)
model = model.to(device, memory_format=torch.channels_last)

try:
    model = torch.compile(model)
    print("Using JIT-compiled model.")
except Exception as e:
    print(f"torch.compile() failed: {e}")
    print("Proceeding without JIT compilation.")

optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

for epoch in range(1, stage1_epochs + 1):
    stage1_loss = train_stage1(epoch, model, dataloader, optimizer, accumulation_steps=accumulation_steps, log_interval=log_interval)
    print(f"Stage1 Epoch {epoch}: Loss: {stage1_loss:.6f}")

model.build_stage1_teacher()

best_loss = float("inf")
for epoch in range(1, stage2_epochs + 1):
    stage2_loss = train_stage2(
        epoch,
        model,
        dataloader,
        optimizer,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        accumulation_steps=accumulation_steps,
        log_interval=log_interval
    )
    print(f"Stage2 Epoch {epoch}: Loss: {stage2_loss:.6f}")

    if stage2_loss < best_loss:
        best_loss = stage2_loss
        torch.save(model.state_dict(), "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/pretrain/best_MCMAE.pth")
        print(f"Saved best model with Loss: {best_loss:.6f}")

torch.save(model.state_dict(), "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/pretrain/MCMAE.pth")
print("Training complete. Model saved.")
