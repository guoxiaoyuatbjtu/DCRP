# DCRP
Official Repository for "DCRP: Class-Aware Feature Diffusion Constraint and Reliable Pseudo-labeling for Imbalanced Semi-Supervised Learning" accepted by IEEE Transactions on Multimedia

## Abstract
In this study, we addressed two key challenges in ISSL: maintaining the reliability of pseudo-labels and ensuring a balanced representation of features. Specifically, we propose a class-aware feature-diffusion constraint and reliable pseudo-labeling (DCRP) framework to address these issues. In the DCRP, we counteract the overconfidence problem of softmax by adding an extra class to the typical K class problem without the need for additional parameters. Moreover, we introduced a flexible class-aware feature diffusion constraint in the feature extractor, promoting a more balanced feature diversity.

## For CIFAR10-LT

```bash
sh runCIFAR10_dcrp.sh
```

## For CIFAR100-LT

```bash
sh runCIFAR100_dcrp.sh
```

## Cite This Work
```bibtex
@ARTICLE{10417792,
  author={Guo, Xiaoyu and Wei, Xiang and Zhang, Shunli and Lu, Wei and Xing, Weiwei},
  journal={IEEE Transactions on Multimedia}, 
  title={DCRP: Class-Aware Feature Diffusion Constraint and Reliable Pseudo-Labeling for Imbalanced Semi-Supervised Learning}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  keywords={Training;Feature extraction;Semisupervised learning;Reliability;Data models;Data augmentation;Tail;semi-supervised learning;class-imbalanced learning;feature diffusion;reliable pseudo-labeling},
  doi={10.1109/TMM.2024.3360704}
}
