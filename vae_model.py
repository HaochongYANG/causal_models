import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Dict, Any, Optional, Tuple


class VAEEncoder(nn.Module):
    """Encoder q_phi(z | Y, X, A)"""
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, y: torch.Tensor, x: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([y, x, a], dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

class VAEDecoder(nn.Module):
    """Decoder p_theta(Y | Z, X, A)"""
    def __init__(self, latent_dim: int, output_dim: int, X_dim: int, A_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_size = latent_dim + X_dim + A_dim
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, x, a], dim=-1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        mu_y = self.fc_output(h)
        return mu_y


class VAE_base(nn.Module):
    def __init__(self, config: Dict[str, Any], device: str):
        super().__init__()
        self.device = device
        
        data_dim = config["model"]["data_dim"] # Total features (182)
        
        # [T, Y0, Y1, Mu0, Mu1, X...]
        self.X_dim = data_dim - 5       # Covariates X
        self.Y_dim = 2                  # Outcomes Y0, Y1 
        self.A_dim = 1                  # Treatment A
        self.latent_dim = config["model"].get("latent_dim", 20)
        
        encoder_input_dim = self.Y_dim + self.X_dim + self.A_dim
        decoder_input_dim = self.latent_dim + self.X_dim + self.A_dim

        self.encoder = VAEEncoder(encoder_input_dim, self.latent_dim)
        self.decoder = VAEDecoder(self.latent_dim, self.Y_dim, self.X_dim, self.A_dim)
        
        self.log_scale = nn.Parameter(torch.zeros(1)) 
        self.use_ipw = config['model']['use_ipw']
        
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def calc_loss(self, observed_data: torch.Tensor, gt_mask: torch.Tensor, propnet: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Computes the weighted Negative ELBO loss L(theta, phi, pi_hat).
        applies masking so we only reconstruct the observed outcome.
        """
        
        # Flatten input: [B, 1, D_total] -> [B, D_total]
        obs_data_flat = observed_data.squeeze(1) 
        mask_flat = gt_mask.squeeze(1)

        # Extract components: [A, Y0, Y1, Mu0, Mu1, ...]
        A = obs_data_flat[:, 0].unsqueeze(1)    # Treatment T [B, 1]
        Y = obs_data_flat[:, 1:1+self.Y_dim]    # Outcomes Y0, Y1 [B, 2]
        X = obs_data_flat[:, 5:]                # Covariates X [B, X_dim]
        
        # this mask is 1 for the observed Y, 0 for the counterfactual.
        Y_mask = mask_flat[:, 1:1+self.Y_dim]   # [B, 2]
        
        # (q_phi(z | Y, X, A))
        Y_encoder_input = Y * Y_mask  # Mask out unobserved outcomes
        mu_z, log_var_z = self.encoder(Y_encoder_input, X, A)
        Z = self.reparameterize(mu_z, log_var_z)

        # p_theta(Y | Z, X, A)
        mu_y_pred = self.decoder(Z, X, A)
        
        scale_var = torch.exp(self.log_scale) # Variance
        
        # Squared Error
        sq_error = torch.pow(Y - mu_y_pred, 2)
        const_term = self.log_scale + np.log(2 * np.pi)
        nll_element = 0.5 * (sq_error / scale_var + const_term)
        
        # Zero out loss for unobserved outcomes
        masked_nll = nll_element * Y_mask
        neg_log_likelihood = torch.sum(masked_nll, dim=1) 
        log_likelihood = -neg_log_likelihood

        kl_element = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp())
        kl_div = torch.sum(kl_element, dim=1)
                
        if self.use_ipw and propnet is not None:
            with torch.no_grad():
                pi_hat = propnet(X.float())
                pi_hat = torch.clamp(pi_hat, min=0.05, max=0.95)
                
                t_batch = A.squeeze(1) 
                
                weights = (t_batch / pi_hat[:, 1]) + ((1 - t_batch) / pi_hat[:, 0])
                
                # clip weights using DiffPO values
                weights = torch.clamp(weights, min=0.5, max=3.0) 
                weights = weights / weights.mean()
        else:
            weights = torch.ones_like(log_likelihood)
        
        elbo = log_likelihood - kl_div
        loss = -torch.mean(weights * elbo)

        return loss

    def forward(self, batch: Dict[str, torch.Tensor], is_train: bool = True, propnet: Optional[nn.Module] = None) -> torch.Tensor:
        """Main forward pass computes the loss."""
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        ) = self.process_data(batch)
        
        loss = self.calc_loss(observed_data, gt_mask, propnet=propnet)
        
        return loss

    def evaluate(self, batch: Dict[str, torch.Tensor], n_samples: int, debug: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Generates N_samples of counterfactual outcomes (Y0, Y1) given X and T=0/T=1.
        """
        
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        ) = self.process_data(batch)
        
        with torch.no_grad():
            
            obs_data_flat = observed_data.squeeze(1)
            X = obs_data_flat[:, 5:]
            
            batch_size = X.size(0)
            
            # P(Z) = N(0, I)
            Z_prior = torch.randn(batch_size * n_samples, self.latent_dim, device=self.device)
            
            X_repeated = X.repeat(n_samples, 1) # [B*nsample, X_dim]
                        
            # A=0 vector (for Y0)
            A0_vector = torch.zeros(batch_size * n_samples, self.A_dim, device=self.device)
            mu_y0_samples_flat = self.decoder(Z_prior, X_repeated, A0_vector) # [B*nsample, Y_dim]

            # A=1 vector (for Y1)
            A1_vector = torch.ones(batch_size * n_samples, self.A_dim, device=self.device)
            mu_y1_samples_flat = self.decoder(Z_prior, X_repeated, A1_vector) # [B*nsample, Y_dim]
                        
            # Reshape back to [B, nsample, Y_dim] where Y_dim=2 ([Y0_pred, Y1_pred])
            y0_pred_samples = mu_y0_samples_flat.view(n_samples, batch_size, self.Y_dim).permute(1, 0, 2)
            y1_pred_samples = mu_y1_samples_flat.view(n_samples, batch_size, self.Y_dim).permute(1, 0, 2)

            y0_final_samples = y0_pred_samples[:,:,0].unsqueeze(-1)
            y1_final_samples = y1_pred_samples[:,:,1].unsqueeze(-1)

            # Final samples shape: [B, nsample, 2] where 2 is (Y0, Y1)
            samples = torch.cat([y0_final_samples, y1_final_samples], dim=-1)
            
        # Return samples along with observed data components for metric calculation
        return samples, observed_data, gt_mask, observed_mask, observed_tp


class VAEPO(VAE_base):
    """
    Main VAE Causal Model class that initializes the VAE_base structure.
    """
    def __init__(self, config: Dict[str, Any], device: str):
        # We pass target_dim=1, but the base class uses config["model"]["data_dim"]
        super(VAEPO, self).__init__(config, device)
    
    def process_data(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Processes data from the DataLoader batch, similar to DiffPO.
        
        Returns: observed_data, observed_mask, observed_tp, gt_mask, for_pattern_mask, cut_length
        """
        observed_data = batch["observed_data"][:, np.newaxis, :]
        observed_data = observed_data.to(self.device).float() # [B, 1, D_total]

        observed_mask = batch["observed_mask"][:, np.newaxis, :]
        observed_mask = observed_mask.to(self.device).float()

        observed_tp = batch["timepoints"].to(self.device).float()

        gt_mask = batch["gt_mask"][:, np.newaxis, :]
        gt_mask = gt_mask.to(self.device).float()

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )