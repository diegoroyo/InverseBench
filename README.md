
**Mapping Dark-Matter Clusters via Physics-Guided Diffusion Models**

Diego Royo, Brandon Zhao, Adolfo Muñoz, Diego Gutierrez, Katherine L. Bouman 

*Abstract:* Galaxy clusters are powerful probes of astrophysics and cosmology through gravitational lensing: the clusters' mass, dominated by 85% dark matter, distorts background light. Yet, mass reconstruction lacks the scalability and large-scale benchmarks to process the hundreds of thousands of clusters expected from forthcoming wide-field surveys. We introduce a fully automated method to reconstruct cluster surface mass density from photometry and gravitational lensing observables. Central to our approach is DarkClusters-15k, our new dataset of 15,000 simulated clusters with paired mass and photometry maps, the largest benchmark to date, spanning multiple redshifts and simulation frameworks. We train a plug-and-play diffusion prior on DarkClusters-15k that learns the statistical relationship between mass and light, and draw posterior samples constrained by weak- and strong-lensing observables; this yields principled reconstructions driven by explicit physics, alongside well-calibrated uncertainties. Our approach requires no expert tuning, runs in minutes rather than hours, achieves higher accuracy, and matches expertly-tuned reconstructions of the MACS 1206 cluster. We release our method and DarkClusters-15k to support development and benchmarking for upcoming wide-field cosmological surveys.

## License
See the original License from the InverseBench project.

## Citation
```bibtex
@article{royo2026mapping,
  author    = {Royo, Diego and Zhao, Brandon and Muñoz, Adolfo and Gutierrez, Diego and Bouman, Katherine L.},
  title     = {Mapping Dark-Matter Clusters Via Physics-Guided Diffusion Models},
  year      = {2026},
}
```