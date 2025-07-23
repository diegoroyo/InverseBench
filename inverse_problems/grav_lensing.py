from abc import ABC, abstractmethod
from torch.autograd import grad
import numpy as np

import torch
from typing import Dict
from .base import BaseOperator


class WeakOnlyGravLensing(BaseOperator):
    def __init__(self,
                 sigma_noise=0.0,
                 image_size=512,
                 use_shear=True,
                 use_flexion=False,
                 unnorm_shift=0.0, unnorm_scale=1.0, device=None):
        self.sigma_noise = sigma_noise
        self.image_size = image_size
        self.use_shear = use_shear
        self.use_flexion = use_flexion
        self.unnorm_shift = unnorm_shift
        self.unnorm_scale = unnorm_scale
        self.device = device

        self.weak_pixels = torch.rand((3000, 2), device=self.device) * self.image_size
        self.noise = torch.randn((6000,), device=self.device) * self.sigma_noise

    def convergence_to_shear(self, kappa):
        import torch
        ny, nx = kappa.shape
        sy, sx = ny * 2 - 1, nx * 2 - 1
        kappa_fft = torch.fft.fft2(kappa, s=(sy, sx))
        
        fx, fy = torch.meshgrid(torch.fft.fftfreq(sy, device=self.device), torch.fft.fftfreq(sx, device=self.device))

        denom = (fx * fx + fy * fy)
        denom[0, 0] = torch.inf
        kernel_1 = (fy * fy - fx * fx) / denom
        kernel_2 = 2 * fx * fy / denom

        g1 = torch.fft.ifft2(kappa_fft * kernel_1).real
        g2 = torch.fft.ifft2(kappa_fft * kernel_2).real
        return g1[:ny, :nx], g2[:ny, :nx]

    def convergence_to_flexion(self, kappa):
        import torch
        ny, nx = kappa.shape
        sy, sx = ny * 2 - 1, nx * 2 - 1
        kappa_fft = torch.fft.fft2(kappa, s=(sy, sx))
        
        fx, fy = torch.meshgrid(torch.fft.fftfreq(sy, device=self.device), torch.fft.fftfreq(sx, device=self.device))

        kernel_1 = -2j * torch.pi * fy
        kernel_2 = -2j * torch.pi * fx
        
        f1 = torch.fft.ifft2(kappa_fft * kernel_1).real
        f2 = torch.fft.ifft2(kappa_fft * kernel_2).real
        return f1[:ny, :nx], f2[:ny, :nx]

    def interpolate_linear(self, x, y, img):
        import torch
        x = torch.maximum(x - 0.5, torch.tensor(0))
        y = torch.maximum(y - 0.5, torch.tensor(0))

        x0, x1 = torch.floor(x).long(), torch.ceil(x).long()
        y0, y1 = torch.floor(y).long(), torch.ceil(y).long()
        x1 = torch.minimum(torch.tensor(img.shape[1] - 1), x1)
        y1 = torch.minimum(torch.tensor(img.shape[0] - 1), y1)
        xt, yt = x - x0, y - y0
        return (
            img[y0, x0] * (1 - xt) * (1 - yt) +
            img[y0, x1] * xt * (1 - yt) +
            img[y1, x0] * (1 - xt) * yt +
            img[y1, x1] * xt * yt
        )

    def forward(self, inputs, **kwargs):
        # inputs has shape (N, C, H, W)
        N, C, H, W = inputs.shape
        assert H == W == 512

        outputs = []
        for i in range(N):
            img = inputs[i, 0]
            # unnormalize the image
            # img = torch.exp(img * 2)

            if self.use_shear:
                g1, g2 = self.convergence_to_shear(img)
                g1 = self.interpolate_linear(self.weak_pixels[:, 0], self.weak_pixels[:, 1], g1)
                g2 = self.interpolate_linear(self.weak_pixels[:, 0], self.weak_pixels[:, 1], g2)
                outputs.append(torch.cat((g1, g2), dim=0))
            if self.use_flexion:
                f1, f2 = self.convergence_to_flexion(img)
                f1 = self.interpolate_linear(self.weak_pixels[:, 0], self.weak_pixels[:, 1], f1)
                f2 = self.interpolate_linear(self.weak_pixels[:, 0], self.weak_pixels[:, 1], f2)
                outputs.append(torch.cat((f1, f2), dim=0))

        result = torch.stack(outputs, dim=0).reshape(N, -1)
        return result + self.noise.reshape(1, -1)
    

class WeakPlusPhotometryGravLensing(WeakOnlyGravLensing):
    def forward(self, inputs, **kwargs):
        weak_obs = super().forward(inputs, **kwargs)
        N = weak_obs.shape[0]
        photo = inputs[:, 1]
        result = torch.cat([weak_obs, photo.reshape(N, -1) / photo.numel()], dim=1)
        return result
    

class StrongOnlyGravLensing(BaseOperator):
    """
    FIXME not well suited for posterior estimation (likelihood gradients will only be
    computed for ~30 pixels corresponding to those that have multiple images)

    Also kinda hacky (data loader stores /tmp/lensed_pixels.pkl file with the results,
    and that is read here)
    """
    def __init__(self, sigma_noise=0.0, unnorm_shift=0.0, unnorm_scale=1.0, device=None):
        self.sigma_noise = sigma_noise
        self.unnorm_shift = unnorm_shift
        self.unnorm_scale = unnorm_scale
        self.device = device

    def image_hash32(self, img: torch.Tensor) -> torch.Tensor:
        """
        Computes a 32-bit integer hash for a 2D image tensor.
        Suitable for torch.int64 dtype input; result is torch.int32 scalar.
        """
        import torch
        assert img.ndim == 2, "Image must be 2D"
        # Ensure consistent integer dtype
        x = img.to(torch.int64).contiguous()
        
        MULT = 6364136223846793005  # LCG multiplier
        INC = 1                     # LCG increment

        # Reduce along the last dimension iteratively
        while x.ndim > 0:
            acc = torch.zeros_like(x[..., 0])
            for i in range(x.shape[-1]):
                acc = acc * MULT + INC + x.select(-1, i)
            x = acc

        # Cast final 64-bit result into 32-bit hash
        return x.to(torch.uint32)

    @torch.no_grad
    def compute_strong_lenses(self, kappa, verbose=False, plot=False, force_write=False):
        from caustics.utils import meshgrid
        from caustics import PixelatedConvergence, FlatLambdaCDM, LensSource
        from matplotlib.lines import Line2D
        import torch
        other_device = torch.get_default_device()
        torch.set_default_device(self.device)
        import pickle
        import os
        import matplotlib.pyplot as plt
        seed = self.image_hash32(kappa)

        if not force_write and os.path.exists(f'/tmp/lensed_pixels.pkl'):
            with open(f'/tmp/lensed_pixels.pkl', 'rb') as f:
                result = pickle.load(f)
                for i, (source, lensed) in enumerate(result):
                    result[i] = (
                        source.to(self.device),
                        lensed.to(self.device),
                    )
                # torch.set_default_device(other_device)
            return result

        fov_x = 10
        fov_y = 10

        cosmology = FlatLambdaCDM()

        lens = PixelatedConvergence(cosmology=cosmology,
                                    pixelscale=fov_x / kappa.shape[0],
                                    z_l=0.3, x0=fov_x/2, y0=fov_y/2,
                                    convergence_map=kappa)

        n_pix = kappa.shape[0]
        res = fov_x / n_pix
        thx, thy = meshgrid(
            res,
            n_pix,
            n_pix,
            dtype=torch.float32,
        )
        thx = thx.to(self.device) + fov_x / 2
        thy = thy.to(self.device) + fov_y / 2

        z_s = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].set_title("Convergence (full lens)")
        axs[1].set_title("Zoom @ critical, caustic lines")
        axs[2].set_title("Zoom @ lensed points")
        cmap = axs[0].imshow(np.log(kappa.detach().cpu().numpy()), cmap='gray', origin='lower', extent=(0, fov_y, 0, fov_x))
        fig.colorbar(cmap, ax=axs[0])
        cmap = axs[1].imshow(np.log(kappa.detach().cpu().numpy()), cmap='gray', origin='lower', extent=(0, fov_y, 0, fov_x))
        fig.colorbar(cmap, ax=axs[1])
        cmap = axs[2].imshow(np.log(kappa.detach().cpu().numpy()), cmap='gray', origin='lower', extent=(0, fov_y, 0, fov_x))
        fig.colorbar(cmap, ax=axs[2])

        A = lens.jacobian_lens_equation(thx, thy, z_s)
        detA = torch.linalg.det(A)

        CS = axs[0].contour(thx.detach().cpu().numpy(), thy.detach().cpu().numpy(), detA.detach().cpu().numpy(), levels=[0.0], colors="b", zorder=1)
        CS = axs[1].contour(thx.detach().cpu().numpy(), thy.detach().cpu().numpy(), detA.detach().cpu().numpy(), levels=[0.0], colors="b", zorder=1)
        CS = axs[2].contour(thx.detach().cpu().numpy(), thy.detach().cpu().numpy(), detA.detach().cpu().numpy(), levels=[0.0], colors="b", zorder=1)
        # Get the path from the matplotlib contour plot of the critical line

        min_x, max_x = None, None
        min_y, max_y = None, None

        paths = CS.allsegs[0]
        axs[1].plot([0], [0], color='r', lw=1, label='Critical line')
        caustic_paths = []
        for i, path in enumerate(paths):
            # Collect the path into a discrete set of points
            x1 = torch.tensor(list(float(vs[0]) for vs in path)).to(self.device)
            x2 = torch.tensor(list(float(vs[1]) for vs in path)).to(self.device)
            # raytrace the points to the source plane
            y1, y2 = lens.raytrace(x1, x2, z_s)
            if len(y1) == 0 or len(y2) == 0:
                continue
            min_x = y1.min() if min_x is None else min(min_x, y1.min())
            max_x = y1.max() if max_x is None else max(max_x, y1.max())
            min_y = y2.min() if min_y is None else min(min_y, y2.min())
            max_y = y2.max() if max_y is None else max(max_y, y2.max())

            # Plot the caustic
            if plot:
                axs[0].plot(y1, y2, color="r", zorder=1)
                axs[1].plot(y1, y2, color="r", zorder=1)
                axs[2].plot(y1, y2, color="r", zorder=1)

        if min_x is None or max_x is None or min_y is None or max_y is None:
            if verbose: print("No critical line found, skipping")
            plt.close(fig)
            return
        x_range = max_x - min_x
        y_range = max_y - min_y
        if x_range < 0.15 or y_range < 0.15:
            if verbose: print(f"Critical line is too small ({x_range:.2f} x {y_range:.2f}), skipping")
            plt.close(fig)
            return
        if x_range > 2.0 or y_range > 2.0:
            if verbose: print(f"Critical line is too large ({x_range:.2f} x {y_range:.2f}), skipping")
            plt.close(fig)
            return
        padding = 0.3
        if plot:
            axs[1].set_xlim(min_x - x_range * padding, max_x + x_range * padding)
            axs[1].set_ylim(min_y - y_range * padding, max_y + y_range * padding)

        np.random.seed(seed)

        target_num_sources = np.random.choice([3,4,5])
        colors = ['g', 'orange', 'purple', 'blue', 'cyan']
        sources = []
        attempts = 0
        while len(sources) < target_num_sources:
            attempts += 1
            if attempts > 200:
                if verbose: print("Too many attempts to find sources, skipping")
                plt.close(fig)
                return
            s_x = torch.tensor(np.random.uniform(min_x.detach().cpu().numpy(), max_x.detach().cpu().numpy())).to(self.device)
            s_y = torch.tensor(np.random.uniform(min_y.detach().cpu().numpy(), max_y.detach().cpu().numpy())).to(self.device)
        
            l_x, l_y = lens.forward_raytrace(s_x, s_y, z_s)
            if len(l_x) <= 1:
                continue

            good = True
            for sp_x, sp_y, _, __ in sources:
                if torch.norm(torch.tensor([s_x, s_y]) - torch.tensor([sp_x, sp_y])) < 0.1:
                    good = False
                    break
            if not good:
                continue

            sources.append((s_x, s_y, l_x, l_y))

        min_x, max_x = None, None
        min_y, max_y = None, None

        result = []
        for i, (s_x, s_y, l_x, l_y) in enumerate(sources):
            min_x = l_x.min() if min_x is None else min(min_x, l_x.min())
            max_x = l_x.max() if max_x is None else max(max_x, l_x.max())
            min_y = l_y.min() if min_y is None else min(min_y, l_y.min())
            max_y = l_y.max() if max_y is None else max(max_y, l_y.max())
            if plot:
                if i == 0:
                    axs[1].scatter(s_x, s_y, color=colors[i], marker='.', s=50, label='Source')
                else:
                    axs[1].scatter(s_x, s_y, color=colors[i], marker='.', s=50)
                for j, (l_xi, l_yi) in enumerate(zip(l_x, l_y)):
                    if i == j == 0:
                        axs[2].scatter(l_xi, l_yi, color=colors[i], marker='x', s=50, label='Lensed source')
                    else:
                        axs[2].scatter(l_xi, l_yi, color=colors[i], marker='x', s=50)
            result.append((
                torch.tensor([s_x, s_y]).to(self.device),
                torch.stack([l_x, l_y], dim=1).to(self.device),
            ))

        x_range = max_x - min_x
        y_range = max_y - min_y
        padding = 0.3
        if plot:
            axs[2].set_xlim(min_x - x_range * padding, max_x + x_range * padding)
            axs[2].set_ylim(min_y - y_range * padding, max_y + y_range * padding)
            # axs[0].legend()
            axs[1].legend()
            axs[2].legend()
        if plot:
            plt.show()
        else:
            plt.close(fig)

        with open(f'/tmp/lensed_pixels.pkl', 'wb') as f:
            pickle.dump(result, f)
        # torch.set_default_device(other_device)
        return result

    def forward(self, inputs, **kwargs):
        from caustics import PixelatedConvergence, FlatLambdaCDM
        # inputs has shape (N, C, H, W)
        N, C, H, W = inputs.shape
        assert H == W == 512
        assert C == 1

        cosmology = FlatLambdaCDM()

        z_s = torch.tensor(1.0).to(self.device)

        outputs = []
        for i in range(N):
            img = inputs[i, 0]
            # undo normalization
            img = (torch.exp(torch.clamp(img, -5, 5) * 2) + (0.018638149 - 1e-1)) / 4

            result = self.compute_strong_lenses(img, verbose=False, plot=False)
            if result is None:
                print('No data????')

            other_device = torch.get_default_device()
            torch.set_default_device(self.device)
            lens = PixelatedConvergence(cosmology=cosmology,
                                    pixelscale=10 / img.shape[0],
                                    z_l=0.3, x0=10/2, y0=10/2,
                                    convergence_map=img)
            

            for source, lensed in result:
                a1, a2 = lens.reduced_deflection_angle(*lensed.T, z_s)
                unlensed = lensed - torch.stack((a1, a2), dim=1)
                for i in range(len(lensed)):
                    for j in range(i+1, len(lensed)):
                        outputs.append(unlensed[i] - unlensed[j])

            # torch.set_default_device(other_device)

        result = torch.stack(outputs, dim=0)
        return result