import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np


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
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        x_next = torch.randn(
            num_samples,
            self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device,
            requires_grad=True)

        pbar = tqdm(range(self.num_steps))

        optimizer = torch.optim.Adam([x_next], lr=self.scale)
        
        for i in pbar:
            optimizer.zero_grad()
            # x_cur = x_next.clone().detach().requires_grad_(True)

            loss = self.forward_op.loss(x_next, observation, conditioning=conditioning, i=i).sum()
            # ll_grad, loss = self.forward_op.gradient(x_next, observation, conditioning=conditioning, return_loss=True, i=i)
            # loss = loss.item()

            loss_tv = 0.0
            if self.tv_regularization_weight > 0:
                loss_tv = self.tv_regularization_weight * torch.mean(torch.sqrt((x_next[:,:,:-1,1:]-x_next[:,:,:-1,:-1])**2 + (x_next[:,:,1:,:-1]-x_next[:,:,:-1,:-1])**2 + 1e-3))
                # tv_grad = torch.autograd.grad(loss_tv, x_next)[0]
                # loss += loss_tv.item()
                # ll_grad += tv_grad

            pbar.set_description(f'Iteration {i + 1}/{self.num_steps}. Data fitting loss: {loss:.3e}, TV loss: {loss_tv:.3e}')
            
            loss += loss_tv
            loss.backward()
            optimizer.step()
            # with torch.no_grad():
            #     x_next -= ll_grad * self.scale
        return x_next