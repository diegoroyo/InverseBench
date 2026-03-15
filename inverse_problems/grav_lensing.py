import torch
import numpy as np
from .base import BaseOperator
from training.dataset import DarkClustersDataset

unnormalize_surface_mass_density = lambda x: DarkClustersDataset.unnormalize_mass(torch.clamp(x, -1, 2))


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


class WeakOnlyGravLensing(BaseOperator):
    """
        Weak lensing forward operator. Compares lensing shear values (gamma).

        For efficiency purposes, instead of converting the surface mass density map
        to a shear field, the GT observations are converted to a convergence map (kappa),
        and the loss is computed in convergence space. Mainly this allows to only do
        one conversion kappa<->gamma and makes loss gradients a bit more stable.
    """

    DEBUG_WL = False

    def __init__(self,
                 image_size=512,
                 sigma_noise=0.0,
                 fov=225,
                 z_l=0.5,
                 shear_filename=None,
                 n_shear_pixels=420,
                 noise_profile='traditional',
                 unnorm_shift=0.0, unnorm_scale=1.0, device=None):
        assert shear_filename is not None, 'You should set shear_filename to load pre-simulated shear data ' \
            'following the same format as in the DarkClusters dataset.'
        self.image_size = image_size
        self.sigma_noise = 0.0
        self.unnorm_shift = unnorm_shift
        self.unnorm_scale = unnorm_scale
        self.device = device
        self.fov = fov
        self.n_shear_pixels = n_shear_pixels
        self.noise_profile = noise_profile

        self.z_l = z_l
        # every WL observation comes from a source at a different redshift
        # the code below rescales all shears to a common reference redshift
        self.z_ref = 1.0
        self.sigma_cr_ref = critical_density(self.z_l, self.z_ref)
        # WL is pretty low res, so instead of computing an image in full
        # resolution (typically 512x512), we do 512/4=128 and then upsample
        self.downscale = 4

        if shear_filename is not None:
            assert sigma_noise == 0.0, "If you're loading shear data, do not add noise on top of it"
            self.shear_data = np.load(shear_filename).astype(np.float32)
        else:
            if sigma_noise == 0.0:
                print('Warning: simulating shear data without noise. '
                      'Keep in mind that this is not realistic.')
            raise NotImplementedError(
                "On-the-fly shear simulation not implemented yet")

    def convergence_to_shear(self, kappa):
        import torch
        ny, nx = kappa.shape
        sy, sx = ny * 2, nx * 2
        kappa_fft = torch.fft.fft2(kappa, s=(sy, sx))

        fx, fy = torch.meshgrid(
            torch.fft.fftfreq(sy, device=self.device),
            torch.fft.fftfreq(sx, device=self.device),
            indexing='ij')

        denom = (fx * fx + fy * fy)
        denom[0, 0] = torch.inf
        kernel_1 = (fy * fy - fx * fx) / denom
        kernel_2 = 2 * fx * fy / denom

        g1 = torch.fft.ifft2(kappa_fft * kernel_1).real
        g2 = torch.fft.ifft2(kappa_fft * kernel_2).real
        return g1[:ny, :nx], g2[:ny, :nx]

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

    def density_estimation_rbf(self, coords, weights, range_x, range_y, noise_profile='traditional'):
        if hasattr(coords, 'detach'):
            coords = coords.detach().cpu().numpy()
        if hasattr(weights, 'detach'):
            weights = weights.detach().cpu().numpy()
        from scipy.interpolate import Rbf
        if noise_profile == 'traditional':
            rbf = Rbf(coords[:, 0], coords[:, 1], weights, function='multiquadric',
                      epsilon=2e-5, smooth=1000000.0)
        elif noise_profile == 'kinematic':
            rbf = Rbf(coords[:, 0], coords[:, 1], weights, function='multiquadric',
                      epsilon=0.1, smooth=0.1)
        elif noise_profile == 'noiseless':
            rbf = Rbf(coords[:, 0], coords[:, 1], weights, function='thin_plate',
                      epsilon=0.1, smooth=0)
        else:
            raise AssertionError(
                "Unknown noise profile for RBF density estimation")

        assert len(coords) == len(weights)

        resolution = 512 // self.downscale

        min_x, max_x = range_x
        min_y, max_y = range_y
        range_x = max_x - min_x
        range_y = max_y - min_y
        x = np.linspace(min_x + range_x / (resolution + 1),
                        max_x - range_x / (resolution + 1), resolution)
        y = np.linspace(min_y + range_y / (resolution + 1),
                        max_y - range_y / (resolution + 1), resolution)
        X, Y = np.meshgrid(x, y, indexing='ij')
        coords = np.stack([X, Y], axis=-1)
        grid_vals = rbf(Y.ravel(), X.ravel()).reshape(X.shape)

        return grid_vals

    @torch.no_grad()
    def compute_dense_shear_from_observations(self, data_idx):
        """ Observations is (N, 5) tensor with (x, y, g1, g2, z_s) """

        weak_pixels = torch.tensor(
            self.shear_data[data_idx, :self.n_shear_pixels, 0:2] * self.image_size / self.fov, device=self.device, dtype=torch.float32)
        shear_values = torch.tensor(
            self.shear_data[data_idx, :self.n_shear_pixels, 2:4], device=self.device, dtype=torch.float32)
        shear_redshifts = torch.tensor(
            self.shear_data[data_idx, :self.n_shear_pixels, 4], device=self.device, dtype=torch.float32)
        crit_densities = []
        shear_redshift_terms = []
        for z_s in shear_redshifts:
            old = critical_density(self.z_l, z_s.item())
            new = self.sigma_cr_ref
            crit_densities.append(old)
            shear_redshift_terms.append(old / new)
        shear_redshift_terms = torch.tensor(
            shear_redshift_terms, device=self.device, dtype=torch.float32)

        g1_gt_sparse, g2_gt_sparse = shear_values[:, 0], shear_values[:, 1]

        # NOTE hera data works much better when you divide
        g1_gt_sparse = g1_gt_sparse * shear_redshift_terms
        g2_gt_sparse = g2_gt_sparse * shear_redshift_terms

        # filter only WL pixels on GT
        g1_gt_dense = self.density_estimation_rbf(
            weak_pixels / self.downscale,
            g1_gt_sparse,
            (0, self.image_size // self.downscale),
            (0, self.image_size // self.downscale),
            noise_profile=self.noise_profile)
        g1_gt_dense = torch.tensor(g1_gt_dense, device=self.device).reshape(
            1, 1, self.image_size // self.downscale, self.image_size // self.downscale)
        g1_gt_dense = torch.nn.functional.interpolate(
            g1_gt_dense, size=(self.image_size, self.image_size), mode='bilinear')
        g2_gt_dense = self.density_estimation_rbf(
            weak_pixels / self.downscale,
            g2_gt_sparse,
            (0, self.image_size // self.downscale),
            (0, self.image_size // self.downscale),
            noise_profile=self.noise_profile)
        g2_gt_dense = torch.tensor(g2_gt_dense, device=self.device).reshape(
            1, 1, self.image_size // self.downscale, self.image_size // self.downscale)
        g2_gt_dense = torch.nn.functional.interpolate(
            g2_gt_dense, size=(self.image_size, self.image_size), mode='bilinear')

        if self.DEBUG_WL:
            def shear_to_convergence(g1, g2):
                """
                Inverts the Fourier-space relation used in `convergence_to_shear`.
                Returns a reconstructed convergence kappa (same shape as inputs).
                Note: the DC (mean) mode of kappa cannot be recovered from shear and will be set to 0.
                """
                ny, nx = g1.shape
                sy, sx = ny * 2 - 1, nx * 2 - 1
                device = g1.device
                dtype = g1.dtype

                # FFT the shears to the padded grid (same s=(sy,sx) used in forward)
                G1 = torch.fft.fft2(g1, s=(sy, sx))
                G2 = torch.fft.fft2(g2, s=(sy, sx))

                # frequency grids (use indexing='ij' so shapes are (sy, sx))
                fy = torch.fft.fftfreq(sy, device=device, dtype=dtype)
                fx = torch.fft.fftfreq(sx, device=device, dtype=dtype)
                fx, fy = torch.meshgrid(fy, fx, indexing='ij')  # note order matches original code

                # build the same kernels as forward (avoid division by zero at [0,0])
                denom = fx * fx + fy * fy
                denom_safe = denom.clone()
                denom_safe[0, 0] = float('inf')

                k1 = (fy * fy - fx * fx) / denom_safe
                k2 = 2.0 * fx * fy / denom_safe

                denom_k = k1 * k1 + k2 * k2
                denom_k[0, 0] = float('inf')

                K_hat = (k1 * G1 + k2 * G2) / denom_k
                K_hat[0, 0] = 0.0

                kappa = torch.fft.ifft2(K_hat).real
                return kappa[:ny, :nx]
            
            kappa = shear_to_convergence(g1_gt_dense.float()[0, 0], g2_gt_dense.float()[0, 0])
            torch.save(kappa, '../_debug/_weak_dense_kappa.pt')

            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(15, 10))
            cmap = axs[0].imshow(g1_gt_dense.detach().cpu().numpy().squeeze())
            fig.colorbar(cmap, ax=axs[0])
            cmap = axs[1].imshow(g2_gt_dense.detach().cpu().numpy().squeeze())
            fig.colorbar(cmap, ax=axs[1])
            cmap = axs[2].imshow(kappa.detach().cpu().numpy().squeeze())
            fig.colorbar(cmap, ax=axs[2])
            fig.suptitle(f'Weak lensing')
            plt.tight_layout()
            plt.savefig(f'../_debug/_weak_dense.png')
            plt.close()

        return g1_gt_dense.float()[0, 0], g2_gt_dense.float()[0, 0]

    def select_data_idx(self, data_idx):
        self.g1_gt_dense, self.g2_gt_dense = \
            self.compute_dense_shear_from_observations(data_idx)
        
    def select_likelihood_idx(self, data_idx):
        pass

    def forward(self, inputs, **kwargs):
        # inputs has shape (N, C, H, W)
        N, C, H, W = inputs.shape
        assert H == W == 512

        sigma = unnormalize_surface_mass_density(
            torch.clone(inputs))

        return sigma / self.sigma_cr_ref

    def loss(self, pred, observation, **kwargs):
        """
            data consistency loss between prediction and given observation
            default as L2 loss (summation over batches)
        Args:
            - pred (torch.tensor): predicted parameters (not measurement), shape (batch_size, ...)
            - observation (torch.tensor): observed data, shape (1, ...)
        Returns:
            - loss (torch.tensor): loss value, shape (batch_size, )
        """
        kappas_est = self.forward(pred)
        N, C, H, W = kappas_est.shape
        # assert N == 1, 'Does not work for batch size > 1 yet'

        loss = []
        for i in range(N):
            kappa_est = kappas_est[i, 0]
            gt_g1_i = self.g1_gt_dense
            gt_g2_i = self.g2_gt_dense

            g1_est, g2_est = self.convergence_to_shear(kappa_est)

            loss.append((torch.nn.functional.mse_loss(gt_g1_i, g1_est) +
                         torch.nn.functional.mse_loss(gt_g2_i, g2_est)))
        return torch.stack(loss).to(self.device)


class StrongOnlyGravLensing(BaseOperator):
    """
        Strong lensing forward operator. Works with multiply-imaged sources,
        and compares their positions and photometry.
    """

    def __init__(self,
                 image_size=512, sigma_noise=0.0,
                 strong_lenses_file=None,
                 z_l=0.5, fov=225,
                 likelihood_reference=None,
                 likelihood_folder=None,
                 unnorm_shift=0.0, unnorm_scale=1.0, device=None):
        assert np.isclose(sigma_noise, 0.0), 'sigma_noise != 0.0 is NYI'
        self.image_size = image_size
        self.sigma_noise = sigma_noise
        self.unnorm_shift = unnorm_shift
        self.unnorm_scale = unnorm_scale
        self.device = device

        self.z_l = z_l
        self.fov = fov
        self.downscale = 4
        self.lambda_geo = 1e-2
        self.lambda_img = 1e-3

        if likelihood_reference is not None:
            assert likelihood_folder is None, 'Provide either likelihood_reference or likelihood_folder, not both'
            self.ll_ref = torch.load(likelihood_reference)
            if isinstance(self.ll_ref, dict):
                self.ll_ref = self.ll_ref['recon']
            self.ll_ref = self.ll_ref.to(device)
        else:
            self.ll_ref = None

        if likelihood_folder is not None:
            assert likelihood_reference is None, 'Provide either likelihood_reference or likelihood_folder, not both'
            self.ll_folder = likelihood_folder
        else:
            self.ll_folder = None

        if strong_lenses_file is not None:
            with open(strong_lenses_file, 'rb') as f:
                import pickle
                self.strong_lenses_full = pickle.load(f)
        else:
            raise NotImplementedError(
                'You must provide a file with pre-simulated strong lensing observables for each image, ' \
                'following the same format as in the DarkClusters dataset. On-the-fly strong lens simulation is NYI.')

    def get_strong_lenses(self, idx):
        result = [None] * len(self.strong_lenses_full[idx])
        for i, (z_s, _, lensed) in enumerate(self.strong_lenses_full[idx]):
            result[i] = (
                True,
                torch.tensor(z_s).to(self.device),
                torch.tensor(lensed).to(self.device),
            )
        return result

    def set_ll_ref(self, ll_ref):
        if ll_ref is None:
            self.ll_ref = None
        else:
            self.ll_ref = ll_ref.to(self.device)
    
    def select_likelihood_idx(self, data_idx):
        if self.ll_folder is not None:
            import os
            ll_path = os.path.join(self.ll_folder, f'result_{data_idx.item()}.pt')
            if not os.path.exists(ll_path):
                raise FileNotFoundError(f'Likelihood reference file not found at {ll_path}')
            self.ll_ref = torch.load(ll_path)['recon'].to(self.device)

    def select_data_idx(self, data_idx):
        self.strong_lenses = self.get_strong_lenses(data_idx)

    def forward(self, inputs, **kwargs):
        from caustics import PixelatedConvergence, Sersic, LensSource, FlatLambdaCDM
        torch.set_default_device(self.device)
        # inputs has shape (N, C, H, W)
        N, C, H, W = inputs.shape
        assert H == W == self.image_size
        assert C == 1

        photometries = kwargs.get('conditioning', None)
        step = kwargs.get('i', None)
        def do_plot(i): return step is not None and step % 50 == 0 and i == 0
        assert photometries is not None, 'photometries (conditioning) must be provided'

        outputs = []
        for i in range(N):
            img = inputs[i, 0]
            photometry = photometries[i, 0]

            if do_plot(i):
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 3, figsize=(20, 6))
                cmap = axs[0].imshow(img.detach().cpu().numpy(), origin='lower', cmap='turbo',
                                     extent=(0, self.fov, 0, self.fov))
                fig.colorbar(cmap, ax=axs[0])
                cmap = axs[1].imshow(photometry.detach().cpu().numpy(), origin='lower', cmap='gray',
                                     extent=(0, self.fov, 0, self.fov),
                                     vmin=torch.quantile(photometry, 0.01).item(), vmax=torch.quantile(photometry, 0.99).item())
                fig.colorbar(cmap, ax=axs[1])

            img_unnorm = unnormalize_surface_mass_density(
                torch.clone(img))

            if self.strong_lenses_full is None:
                raise NotImplementedError(
                    'On-the-fly strong lens simulation is NYI')
            else:
                curr_strong_lenses = self.strong_lenses

            oo = torch.linspace(0, self.fov, img_unnorm.shape[0])
            Y, X = torch.meshgrid(oo, oo, indexing='ij')

            loss_geo = 0.0
            loss_img = 0.0
            for lens_idx, (use_image, z_s, images) in enumerate(curr_strong_lenses):
                w = 1 / critical_density(self.z_l, z_s)

                # Note: need to create new cosmology each time to avoid memory leaks in caustics
                cosmology = FlatLambdaCDM()
                lens = PixelatedConvergence(cosmology=cosmology,
                                            pixelscale=self.fov /
                                            img_unnorm.shape[0],
                                            z_l=self.z_l, x0=self.fov/2, y0=self.fov/2,
                                            convergence_map=img_unnorm * w)

                ax, ay = lens.reduced_deflection_angle(*images.T, z_s)
                unlensed = images - torch.stack((ax, ay), dim=1)
                pos_estimate = torch.mean(unlensed, axis=0)

                loss_geo = loss_geo + \
                    torch.mean(torch.linalg.norm(
                        unlensed - pos_estimate.reshape(1, 2), dim=1) ** 2)

                if not use_image:
                    continue

                sersic = Sersic(
                    x0=pos_estimate[0], y0=pos_estimate[1], q=1.0, phi=0.0, n=1, Re=1.0, Ie=1.0)
                system = LensSource(lens=lens, source=sersic, z_s=z_s, x0=self.fov/2, y0=self.fov/2,
                                    pixelscale=self.fov * self.downscale /
                                    img_unnorm.shape[0],
                                    pixels_x=img_unnorm.shape[0] // self.downscale)
                sersic_img = torch.nn.functional.interpolate(
                    system().unsqueeze(0).unsqueeze(0),
                    size=img_unnorm.shape,
                    mode='bilinear').squeeze()
                del sersic, system, lens, cosmology

                if do_plot(i):
                    axs[0].scatter(pos_estimate[0].detach().cpu().numpy(), pos_estimate[1].detach().cpu().numpy(),
                                   marker='x')
                    axs[0].scatter(*(images.detach().cpu().numpy().T),
                                   marker='o')
                    if lens_idx == 4:
                        cmap = axs[2].imshow(sersic_img.detach().cpu().numpy(), origin='lower', cmap='gray',
                                             extent=(0, self.fov, 0, self.fov),
                                             vmin=torch.quantile(
                                                 photometry, 0.01).item(),
                                             vmax=torch.quantile(photometry, 0.99).item())
                        fig.colorbar(cmap, ax=axs[2])
                        for lx, ly in images:
                            axs[2].scatter(lx.detach().cpu().numpy(), ly.detach().cpu().numpy(),
                                           marker='x', color='r')

                with torch.no_grad():
                    mask = torch.zeros(
                        img_unnorm.shape, dtype=torch.bool, device=self.device)
                    s = 1.0
                    for lx, ly in images:
                        mask[(X - lx) ** 2 + (Y - ly) ** 2 <= s ** 2] = 1

                    reference_img = torch.clone(photometry).detach()
                    ref_min = reference_img.min()
                    ref_max = reference_img.max()
                    reference_img = 10.0 * (reference_img - ref_min) / (ref_max - ref_min + 1e-8)
                    # reference_img = reference_img * 10.0 / reference_img.max()

                    prop = (torch.sum(mask) /
                            (mask.shape[0] * mask.shape[1])).detach().item()
                loss_img = loss_img + \
                    torch.mean(
                        (sersic_img[mask] - reference_img[mask]) ** 2) * (1 - prop) * 10
                loss_img = loss_img + \
                    torch.mean(sersic_img[~mask] ** 2) * prop / 10

            loss_geo = loss_geo * self.lambda_geo
            loss_img = loss_img * self.lambda_img
            outputs.append(loss_geo + loss_img)

            if do_plot(i):
                fig.suptitle(
                    f'Geo loss {loss_geo.item():.3e}, Img loss {loss_img.item():.3e}')
                plt.tight_layout()
                plt.savefig(
                    f'../_debug/current_strong_{step}.png')
                plt.close()
        return torch.stack(outputs).reshape(N, 1).to(self.device)

    def loss(self, pred, observation, **kwargs):
        """
            data consistency loss between prediction and given observation
            default as L2 loss (summation over batches)
        Args:
            - pred (torch.tensor): predicted parameters (not measurement), shape (batch_size, ...)
            - observation (torch.tensor): observed data, shape (1, ...)
        Returns:
            - loss (torch.tensor): loss value, shape (batch_size, )
        """
        if self.ll_ref is not None:
            return (50 * (self.ll_ref.unsqueeze(0).unsqueeze(0) - pred) ** 2).flatten(start_dim=1).mean(dim=1)
        else:
            return (self.forward(pred, **kwargs)).flatten(start_dim=1).sum(dim=1)


class WeakAndStrongGravLensing(BaseOperator):
    """ Just calls the two operators above """

    def __init__(self,
                 loss_factor_weak=0.05,  # common parameters
                 loss_factor_strong=1.0,
                 image_size=512,
                 sigma_noise_weak=0.0,
                 sigma_noise_strong=0.0,
                 z_l=0.5,
                 fov=225,
                 shear_filename=None,  # weak-only parameters
                 n_shear_pixels=420,
                 noise_profile='traditional',
                 strong_lenses_file=None,  # strong-only parameters
                 likelihood_reference=None,
                 likelihood_folder=None,
                 unnorm_shift=0.0, unnorm_scale=1.0, device=None):  # final common parameters
        self.loss_factor_weak = loss_factor_weak
        self.loss_factor_strong = loss_factor_strong
        self.sigma_noise = 0.0
        self.unnorm_shift = unnorm_shift
        self.unnorm_scale = unnorm_scale
        self.device = device
        self.weak = WeakOnlyGravLensing(
            image_size=image_size,
            sigma_noise=sigma_noise_weak,
            z_l=z_l,
            shear_filename=shear_filename,
            n_shear_pixels=n_shear_pixels,
            noise_profile=noise_profile,
            unnorm_shift=unnorm_shift, unnorm_scale=unnorm_scale, device=device)
        self.strong = StrongOnlyGravLensing(
            image_size=image_size,
            sigma_noise=sigma_noise_strong,
            strong_lenses_file=strong_lenses_file,
            z_l=z_l,
            fov=fov,
            likelihood_reference=likelihood_reference,
            likelihood_folder=likelihood_folder,
            unnorm_shift=unnorm_shift, unnorm_scale=unnorm_scale, device=device)

    def set_ll_ref(self, ll_ref):
        if ll_ref is None:
            self.strong.ll_ref = None
        else:
            self.strong.ll_ref = ll_ref.to(self.device)

    def select_data_idx(self, data_idx):
        self.weak.select_data_idx(data_idx)
        self.strong.select_data_idx(data_idx)

    def select_likelihood_idx(self, data_idx):
        self.weak.select_likelihood_idx(data_idx)
        self.strong.select_likelihood_idx(data_idx)

    def forward(self, inputs, **kwargs):
        """ NOTE: this forward is not used for loss, or in inference in general. It's just for completeness """
        result_weak = self.weak.forward(inputs, **kwargs)
        result_strong = self.strong.forward(inputs, **kwargs)

        result = torch.cat([result_weak.flatten(start_dim=1),
                           result_strong.flatten(start_dim=1)], dim=1)
        return result

    def loss(self, pred, observation, **kwargs):
        N, C, H, W = pred.shape
        loss_weak = self.weak.loss(pred, observation[:, :N*C*H*W], **kwargs)
        loss_strong = self.strong.loss(
            pred, observation[:, N*C*H*W:], **kwargs)
        return loss_weak * self.loss_factor_weak + loss_strong * self.loss_factor_strong
