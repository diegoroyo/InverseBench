import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
from algo.likelihood import Likelihood
import numpy as np

# -----------------------------------------------------------------------------------------------
# Paper: Diffusion Posterior Sampling for General Noisy Inverse Problems
# Official implementation: https://github.com/DPS2022/diffusion-posterior-sampling
# -----------------------------------------------------------------------------------------------


class DPSAdam(Algo):
    
    '''
    Use already-optimized image as likelihood term.
    '''
    
    def __init__(self, 
                 net,
                 forward_op,
                 learning_rate,
                 num_steps,
                 tv_regularization_weight,
                 diffusion_scheduler_config,
                 sde=True):
        super(DPSAdam, self).__init__(net, forward_op)
        self.likelihood = Likelihood(self.net, self.forward_op,
                                     learning_rate=learning_rate,
                                     tv_regularization_weight=tv_regularization_weight,
                                     num_steps=num_steps)
        self.scale = learning_rate
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde
        
    def inference(self, observation, num_samples=1, conditioning=None, **kwargs):
        import os
        if os.path.exists('/home/droyo/code-darkmatter/_debug/current_ll_ref.pt'):
            print("Loading cached likelihood reference...")
            ll_ref = torch.load('/home/droyo/code-darkmatter/_debug/current_ll_ref.pt')
        else:
            print("Computing likelihood reference...")
            ll_ref = self.likelihood.inference(observation, num_samples=num_samples, conditioning=conditioning, **kwargs)
            torch.save(ll_ref / 1.75, '/home/droyo/code-darkmatter/_debug/current_ll_ref.pt')

        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True

        pbar = tqdm(range(self.scheduler.num_steps))
        
        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)

            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]
            
            denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device), conditioning=conditioning)

            loss = torch.mean((denoised - ll_ref) ** 2)
            ll_grad = torch.autograd.grad(loss, x_cur)[0]

            if i % 50 == 0:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 2, figsize=(18,6))
                cmap = axs[0].imshow(denoised[0,0].cpu().detach().numpy(), cmap='turbo')
                fig.colorbar(cmap, ax=axs[0])
                cmap = axs[1].imshow(ll_ref[0,0].cpu().detach().numpy(), cmap='turbo')
                fig.colorbar(cmap, ax=axs[1])
                plt.savefig(f'/home/droyo/code-darkmatter/_debug/dps_adam_denoised_step_{i}.png')
                plt.close()
            # gradient, loss_scale = self.forward_op.gradient(denoised, observation, conditioning=conditioning, i=i, return_loss=True)

            # ll_grad = torch.autograd.grad(denoised, x_cur, gradient)[0]
            # ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale)

            score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Data fitting loss: {loss}')
            
            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5 
            x_next -= ll_grad * self.scale
        return x_next


# class DPSAdam(Algo):
    
#     '''
#     DPS algorithm implemented in EDM framework.
#     '''
    
#     def __init__(self, 
#                  net,
#                  forward_op,
#                  diffusion_scheduler_config,
#                  guidance_scale,
#                  sde=True):
#         super(DPSAdam, self).__init__(net, forward_op)
#         self.scale = guidance_scale
#         self.diffusion_scheduler_config = diffusion_scheduler_config
#         self.scheduler = Scheduler(**diffusion_scheduler_config)
#         self.sde = sde
        
#     def inference(self, observation, num_samples=1, conditioning=None, **kwargs):
#         device = self.forward_op.device
#         if num_samples > 1:
#             observation = observation.repeat(num_samples, 1, 1, 1)
#         x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution, self.net.img_resolution, device=device) * self.scheduler.sigma_max
        
#         x_cur = x_initial
#         x_cur.requires_grad = True
#         optimizer = torch.optim.Adam([x_cur], lr=self.scale)

#         pbar = tqdm(range(self.scheduler.num_steps))
        
#         for i in pbar:
#             # lr_scale = 1 if i > 300 else (i / 300) ** 2
#             # for g in optimizer.param_groups:
#             #     g['lr'] = self.scale * lr_scale
#             optimizer.zero_grad()

#             sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], self.scheduler.scaling_factor[i]

#             denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device), conditioning=conditioning)
#             # denoised = torch.clamp(denoised, 0.0, 1.0) * denoised_scale

#             loss = self.forward_op.loss(denoised, observation, conditioning=conditioning, i=i).sum()
#             loss_tv = 1e3 * torch.mean(torch.sqrt((x_cur[:,:,:-1,1:]-x_cur[:,:,:-1,:-1])**2 + (x_cur[:,:,1:,:-1]-x_cur[:,:,:-1,:-1])**2 + 1e-3))

#             loss += loss_tv
#             if i > 300:
#                 loss.backward()
#                 original = x_cur.detach().clone()
#                 optimizer.step()
#                 print(torch.sum(torch.abs(original - x_cur.detach().clone())))

#             with torch.no_grad():
#                 score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
                
#                 if self.sde:
#                     epsilon = torch.randn_like(x_cur)
#                     x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
#                 else:
#                     x_next = x_cur * scaling_factor + factor * score * 0.5 

#             optimizer.param_groups[0]['params'][0] = x_next
#             x_cur = x_next
#             x_cur.requires_grad = True

#             pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Data fitting loss: {loss.item():.3e}')
#         return x_cur