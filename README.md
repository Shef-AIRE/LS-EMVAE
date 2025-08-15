# [Multimodal Latent Fusion of ECG Leads for Early Assessment of Pulmonary Hypertension](https://arxiv.org/abs/2503.13470)

## Introduction
This repository contains the PyTorch implementation of the **LS-EMVAE** framework. LS-EMVAE is a lead-specific electrocardiogram multimodal variational autoencoder model to pre-train on a large unlabeled 12L-ECG dataset and transfer its latent representations to fine-tune on smaller task-specific labeled 6L-ECG datasets. Unlike existing approaches that consider multi-lead ECG as a single modality, \textsc{LS-EMVAE} treats each lead as a separate modality. It uses a hierarchical modality expert (HiME) fusion mechanism comprising a mixture of experts and a product of experts to enable flexible, adaptive latent fusion. \textsc{LS-EMVAE} also introduces a latent representation alignment loss to improve coherence between individual leads and shared representations. We validate \textsc{LS-EMVAE} across two retrospective cohorts in a 6L-ECG setting: $892$ subjects from the ASPIRE registry for (1) PH detection and (2) phenotyping pre-/post-capillary PH, and $16,416$ subjects from UK Biobank for (3) predicting elevated pulmonary atrial wedge pressure, where it consistently outperforms baseline methods and demonstrates strong generalizability and interpretability.

## Framework
![LS-EMVAE](image/LS-EMVAE.png)

## System Requirements
The source code developed in Python 3.11 using PyTorch 2.3. **LSEMVAE** has been pre-trained on an NVDIA A100 NVLink 80GB GPU with 256GB RAM.


## Datasets and Pre-trained models
The `datasets` for pre-training can be accessed through PhysioNet. MIMIC-IV-ECG is an open-access dataset, and you can find it [here](https://physionet.org/content/mimic-iv-ecg/1.0/).

The in-house ASPIRE registry dataset contains sensitive patient data which cannot be made public because of the General Data Protection Regulation (GDPR).

## References
    [1] Wu, M., Goodman, N.: Multimodal generative models for scalable weaklysupervised learning. Advances in Neural Information Processing Systems 31 (2018)
    [2] M. N. Suvon, P. C. Tripathi, W. Fan, S. Zhou, X. Liu, S. Alabed, V. Osmani, A. J. Swift, C. Chen, and H. Lu, “Multimodal variational autoencoder for low-cost cardiac hemodynamics instability detection,” in International Conference on Medical Image Computing and Computer- Assisted Intervention. Springer, 2024, pp. 296–306.
    [3] Garg, P., Gosling, R., Swoboda, P., Jones, R., Rothman, A., Wild, J.M., Kiely, D.G., Condliffe, R., Alabed, S., Swift, A.J.: Cardiac magnetic resonance identifies raised left ventricular filling pressure: prognostic implications. European Heart Journal 43(26), 2511–2522 (2022)
    [4] K. McKeen, L. Oliva, S. Masood, A. Toma, B. Rubin, and B. Wang, “Ecg-fm: An open electrocardiogram foundation model,” arXiv preprint arXiv:2408.05178, 2024.
    [5] A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” in Forty-first International Confer- ence on Machine Learning, 2024.
