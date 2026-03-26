# Auto-Differentiation Based MRI Pulse Design

This repository is a fork of [tianrluo/AutoDiffPulses](https://github.com/tianrluo/AutoDiffPulses.git), extended to support **steady-state sequence optimization**. This work has been submitted as a technical note to Magnetic Resonance in Medicine (MRM, under review).

Reference implementation of the original work:\
[Joint Design of RF and Gradient Waveforms via Auto-Differentiation for 3D Tailored Exitation in MRI](https://ieeexplore.ieee.org/document/9439482)\
(arXiv: [https://arxiv.org/abs/2008.10594](https://arxiv.org/abs/2008.10594))

cite as:

```bib
@article{luo2021joint,
  author={Luo, Tianrui and Noll, Douglas C. and Fessler, Jeffrey A. and Nielsen, Jon-Fredrik},
  journal={IEEE Transactions on Medical Imaging},
  title={Joint Design of RF and gradient waveforms via auto-differentiation for 3D tailored excitation in MRI},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2021.3083104}}
```

For the `interpT` feature, consider citing:
```bib
@inproceedings{luo2021MultiScale,
  title={Multi-scale Accelerated Auto-differentiable Bloch-simulation based joint design of excitation RF and gradient waveforms},
  booktitle={ISMRM},
  pages={3958},
  author={Tianrui Luo and Douglas C. Noll and Jeffrey A. Fessler and Jon-Fredrik Nielsen},
  year={2021}
}
```

## System Requirements:
- Ubuntu 18.04, 20.04
- Python 3.6, 3.7, 3.8

The implementation was not tested with other configurations.

## General comments

`setup_AutoDiffPulses.m` does the configurations for Matlab.\
For the python part, in your command line, navigate to the repo's root directory, type:

```sh
pip install -e .
```

Demos are provided in `./demo`.

This repo has included binary test data files for basic accessibility in certain regions.\
Future binary data files will be added to: <https://drive.google.com/drive/folders/1EyKhA_d74OC4KADMuTd1kRTEMoVqWdIY>.

## Dependencies

This work requires Python (`≥v3.5`), PyTorch (`≥v1.3`) with CUDA.

- `mrphy_ss`: Python, Github [link](https://github.com/YongliHe23/mrphy_ss.git) — replaces the original `MRphy`, extended with steady-state Bloch simulation support.
- `+mrphy_ss`: Matlab, Github [link](https://github.com/YongliHe23/mrphy_ss.mat.git) — replaces the original `+mrphy`.
- `+attr`: Matlab, Github [link](https://github.com/fmrilab/attr.mat).

Other Python dependencies include:\
`scipy`, `numpy`, `PyTorch`.
