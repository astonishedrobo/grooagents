import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, categorical_dims=None):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        modules = []
        in_features = input_dim
        
        # Build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_features = h_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        modules = []
        hidden_dims.reverse()
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0])
        
        # Build decoder
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        
        self.decoder = nn.Sequential(*modules)
        
        # Final layer
        self.final_layer = nn.Linear(hidden_dims[-1], input_dim)
        
    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var
    
    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
        
    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss

class MaskedBetaVAE(VAE):
    def __init__(self, input_dim, hidden_dims, latent_dim, beta, categorical_dims=None):
        super().__init__(input_dim, hidden_dims, latent_dim, categorical_dims)
        self.beta = beta
        self.register_buffer('beta_mask', torch.ones(latent_dim))
    
    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
        masked_kld_loss = (kld_loss * self.beta_mask).sum()
        total_loss = recon_loss + self.beta * masked_kld_loss
        return total_loss / x.size(0)

    def update_mask(self, mask):
        assert mask.shape == self.beta_mask.shape
        self.beta_mask.data = mask.to(self.beta_mask.device)