import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, latent_dim * 2)

    def forward(self, X):
        X = self.embedding(X)
        X = F.relu(X)
        X = self.hidden_layer(X)
        X = F.relu(X)
        X = self.output_layer(X)
        mu, var = torch.chunk(X, 2, dim=1)
        return mu, var


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.intput_layer = nn.Linear(latent_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        X = self.intput_layer(X)
        X = F.relu(X)
        X = self.hidden_layer(X)
        X = F.relu(X)
        X = self.output_layer(X)
        X = F.sigmoid(X)
        return X


class VAE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(output_dim, hidden_dim, latent_dim)

    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X):
        mu, logvar = self.encoder(X)
        z = self.reparameterize(mu, logvar)
        X = self.decoder(z)
        return X, mu, logvar
