import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


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

class ECGDataset(Dataset):
    def __init__(self, ecg_leads):
        self.ecg_leads = ecg_leads

    def __len__(self):
        return len(next(iter(self.ecg_leads.values())))

    def __getitem__(self, idx):
        # Stack all 12 leads into a single tensor with shape (12, 5000)
        stacked_leads = torch.stack([self.ecg_leads[lead][idx] for lead in self.ecg_leads])
        return stacked_leads  # Shape: (12, 5000)

# Initialize the dataset and dataloader
dataset = ECGDataset(ecg_lead_tensors)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

class BetaVAE(nn.Module):
    def __init__(self, in_channels=12, input_length=5000, latent_dim=256, hidden_dims=None, beta=4, gamma=1000.0, max_capacity=25, Capacity_max_iter=1e5, loss_type='B'):
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.num_iter = 0
        self.input_length = input_length

        # --- Encoder ---
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Calculate encoded shape dynamically
        with torch.no_grad():
            sample_input = torch.zeros(1, 12, input_length)
            encoded_output = self.encoder(sample_input)
            self.encoded_channels = encoded_output.size(1)
            self.encoded_length = encoded_output.size(2)

        # Fully connected layers for the latent space
        self.fc_mu = nn.Linear(self.encoded_channels * self.encoded_length, latent_dim)
        self.fc_var = nn.Linear(self.encoded_channels * self.encoded_length, latent_dim)

        # --- Decoder ---
        self.decoder_input = nn.Linear(latent_dim, self.encoded_channels * self.encoded_length)

        hidden_dims.reverse()
        decoder_modules = []
        for i in range(len(hidden_dims) - 1):
            decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*decoder_modules)
        
        # Calculate required output_padding to exactly match 5000 samples
        self.final_layer = nn.ConvTranspose1d(hidden_dims[-1], 12, kernel_size=4, stride=2, padding=1)
        self.output_length_correction = self._calculate_output_length_correction()

    def _calculate_output_length_correction(self):
        """Calculate how many samples are needed to match 5000."""
        with torch.no_grad():
            sample_input = torch.zeros(1, 12, self.input_length)
            encoded_output = self.encoder(sample_input)
            decoded_output = self.final_layer(self.decoder(encoded_output))
            current_length = decoded_output.size(2)
            return self.input_length - current_length

    def encode(self, input):
        result = self.encoder(input)
        result = result.view(result.size(0), -1)  # Flatten the result
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(z.size(0), self.encoded_channels, self.encoded_length)
        result = self.decoder(result)

        # Final layer with correction applied if necessary
        result = self.final_layer(result)
        if self.output_length_correction != 0:
            result = F.pad(result, (0, self.output_length_correction))  # Correct the output size if needed
        return result

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(self, recons, input, mu, log_var, M_N):
        self.num_iter += 1  # Increment iteration count

        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)

        if self.loss_type == 'H':
            loss = recons_loss + self.beta * M_N * kld_loss
        elif self.loss_type == 'B':
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * M_N * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

print("done loading data and model")


# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BetaVAE(in_channels=12, input_length=5000, latent_dim=256).to(device)

# Try enabling JIT Compilation (torch.compile()), fallback if not supported
try:
    model = torch.compile(model)  # JIT optimization for PyTorch 2.0+
    print("Using JIT-compiled model.")
except Exception as e:
    print(f"torch.compile() failed: {e}")
    print("Proceeding without JIT compilation.")

# Use AdamW optimizer for better performance with large batches
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

# Enable AMP (Automatic Mixed Precision)
scaler = GradScaler()

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Initialize data loaders with improved performance settings
batch_size = 1024  # Increased for speed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Early stopping parameters
patience = 10
best_val_loss = float('inf')
early_stop_counter = 0

# Training loop
epochs = 100
accumulation_steps = 2  # Reduce memory footprint

for epoch in range(epochs):
    # --- Training Phase ---
    model.train()
    total_train_loss = 0

    with tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}] - Training") as progress_bar:
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.float().to(device)

            optimizer.zero_grad()

            with autocast():
                # Forward pass
                recons, mu, log_var = model(batch)

                # Compute loss
                loss_dict = model.loss_function(recons, batch, mu, log_var, M_N=1.0)
                loss = loss_dict['loss'] / accumulation_steps  # Normalize for accumulation

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Gradient clipping
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Average Training Loss: {avg_train_loss:.4f}")

    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.float().to(device)
            with autocast():
                recons, mu, log_var = model(batch)
                loss_dict = model.loss_function(recons, batch, mu, log_var, M_N=1.0)
                total_val_loss += loss_dict['loss'].item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")

    # --- Early Stopping ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "/mnt/parscratch/users/ac1xms/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/data_feature/pretrain/beta_VAE.pth")
        print("Validation loss improved. Model saved.")
    else:
        early_stop_counter += 1
        print(f"No improvement in validation loss. Early stop counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered.")
        break

print("Training completed.")


