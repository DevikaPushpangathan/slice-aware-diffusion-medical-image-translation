import torch

def add_noise(x0, t, alpha_hats):
    noise = torch.randn_like(x0)
    alpha_hat_t = alpha_hats[t].view(-1, 1, 1, 1)
    noisy = torch.sqrt(alpha_hat_t) * x0 + torch.sqrt(1 - alpha_hat_t) * noise
    return noisy, noise
