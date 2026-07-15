> [!WARNING]
> **Work in Progress**
>
> This project is under active development. It is currently **not functional** and should not be used yet.

# Advanced HTR-based system for indexing genealogical scans

Using this tool you can easily index your genealogical scans without the need for human supervision.

The goal of this project is to achieve over 95% accuracy on names and surnames recognition.

![Project Overview](./Project%20Overview.drawio.svg)

## ⏳ Progress

- [x] Data downloader
- [x] Autoencoder (AE)
  - [x] Training dataset
  - [x] Architecture implementation
  - [ ] Model training
- [ ] Record splitter
  - [ ] Training dataset
  - [ ] Architecture implementation
  - [ ] Model training
- [ ] HTR Model
  - [ ] Training dataset
  - [ ] Model selection
  - [ ] Model fine-tuning

## 📜 Refereces

1. Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick, 
*Masked Autoencoders Are Scalable Vision Learners*, https://arxiv.org/abs/2111.06377

2. Yanghao Li, Hanzi Mao, Ross Girshick, Kaiming He,
*Exploring Plain Vision Transformer Backbones for Object Detection*, https://arxiv.org/abs/2203.16527