import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np

# -----------------------------------------------------------------------------------------------
# Paper: Diffusion Posterior Sampling for General Noisy Inverse Problems
# Official implementation: https://github.com/DPS2022/diffusion-posterior-sampling
# -----------------------------------------------------------------------------------------------


class DPSAdam(Algo):
    
    '''
    DPS algorithm implemented in EDM framework.
    '''
    
    def __init__(self, 
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale,
                 sde=True):
        super(DPSAdam, self).__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde
        
    def inference(self, observation, num_samples=1, conditioning=None, **kwargs):
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        
        x_cur = x_initial
        x_cur.requires_grad = True
        optimizer = torch.optim.Adam([x_cur], lr=self.scale)

        pbar = tqdm(range(self.scheduler.num_steps))
        
        for i in pbar:
            # lr_scale = 1 if i > 300 else (i / 300) ** 2
            # for g in optimizer.param_groups:
            #     g['lr'] = self.scale * lr_scale
            optimizer.zero_grad()

            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]

            denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device), conditioning=conditioning)
            denoised_scale = 1 if i > 300 else (i / 300) ** 2
            denoised = denoised * denoised_scale
            # denoised = torch.clamp(denoised, 0.0, 1.0) * denoised_scale

            loss = self.forward_op.loss(denoised, observation, conditioning=conditioning, i=i).sum()
            loss_tv = 1e3 * torch.mean(torch.sqrt((x_cur[:,:,:-1,1:]-x_cur[:,:,:-1,:-1])**2 + (x_cur[:,:,1:,:-1]-x_cur[:,:,:-1,:-1])**2 + 1e-3))

            loss += loss_tv
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
                
                if self.sde:
                    epsilon = torch.randn_like(x_cur)
                    x_cur = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
                else:
                    x_cur = x_cur * scaling_factor + factor * score * 0.5 

            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Data fitting loss: {loss.item():.3e}')
        return x_cur