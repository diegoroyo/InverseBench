import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np

def critical_density(z_l, z_s):
    if hasattr(z_l, 'device'):
        z_l = z_l.detach().cpu().numpy()
    if hasattr(z_s, 'device'):
        z_s = z_s.detach().cpu().numpy()
    from astropy.constants import c, G
    from astropy.cosmology import Planck18 as cosmo
    D_s = cosmo.angular_diameter_distance(z_s)
    D_d = cosmo.angular_diameter_distance(z_l)
    D_ds = cosmo.angular_diameter_distance_z1z2(z_l, z_s)
    crit_density = (c**2 * D_s) / ((4 * np.pi * G * D_ds * D_d) * 10**10)
    crit_density = crit_density.to_value('Msun / (kpc kpc)')
    # convert to Msun h^{-1} / (ckpc/h)^2
    crit_density = crit_density / (cosmo.h * (1 + z_l)**2)
    return crit_density


class Likelihood(Algo):
    '''
    Image-space optimization only with the likelihood term.
    '''
    
    def __init__(self, 
                 net,
                 forward_op,
                 learning_rate,
                 tv_regularization_weight=0.0,
                 num_steps=1000):
        super(Likelihood, self).__init__(net, forward_op)
        self.scale = learning_rate
        self.num_steps = num_steps
        self.tv_regularization_weight = tv_regularization_weight

    def inference(self, observation, num_samples=1, conditioning=None, **kwargs):
        from training.dataset import DarkClustersDataset
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        x_next = torch.ones(
            num_samples,
            self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device,
            dtype=torch.float32) * DarkClustersDataset.ZERO_VALUE
        x_next = x_next.detach().requires_grad_(True)

        unnormalize_surface_mass_density = lambda x: DarkClustersDataset.unnormalize_mass(x)

        pbar = tqdm(range(self.num_steps))

        optimizer = torch.optim.Adam([x_next], lr=self.scale)
        
        for i in pbar:
            optimizer.zero_grad()

            loss = self.forward_op.loss(x_next, observation, conditioning=conditioning, i=i).sum()

            loss_tv = 0.0
            if self.tv_regularization_weight > 0:
                loss_tv = self.tv_regularization_weight * torch.mean(torch.sqrt((x_next[:,:,:-1,1:]-x_next[:,:,:-1,:-1])**2 + (x_next[:,:,1:,:-1]-x_next[:,:,:-1,:-1])**2 + 1e-3))

            pbar.set_description(f'Iteration {i + 1}/{self.num_steps}. Data fitting loss: {loss:.3e}, TV loss: {loss_tv:.3e}')

            # if i % 50 == 0:
            #     sigma_cr = critical_density(0.5, 1.0)
            #     import matplotlib.pyplot as plt
            #     fig, axs = plt.subplots(1, 2, figsize=(18, 6))
            #     cmap = axs[0].imshow(
            #         observation[:, :512*512].reshape(1, 1, 512, 512)[0, 0].cpu().detach().numpy(), cmap='turbo', vmax=0.5)
            #     fig.colorbar(cmap, ax=axs[0])
            #     kappa_est = unnormalize_surface_mass_density(x_next[0, 0]) / sigma_cr.item()
            #     print(kappa_est.max(), kappa_est.min())
            #     cmap = axs[1].imshow(
            #         kappa_est.cpu().detach().numpy(), cmap='turbo', vmax=0.5)
            #     fig.colorbar(cmap, ax=axs[1])
            #     plt.savefig(
            #         f'/home/droyo/code-darkmatter/_debug/_ll_step_{i}.png')
            #     plt.close()
            
            loss += loss_tv
            loss.backward()
            optimizer.step()
        return x_next