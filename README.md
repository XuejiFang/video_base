# Diffusion Codebase
## 1. Implement of CogVideoX-0.6B
### 1.1. Text-to-Image
**Training Recipe**
```yaml
dataset:    ~/phase3_t2v_v4.3_joint_youtube_image/image-9m.pkl
batch_size: 768
lr:         0.0001
warmup:     1000
ema_decay:  0.9999
```
**Loss Curve**

**GenEval**


## ToDo
- [x] [15/1/25] Training Loop
- [ ] [17/1/25] Train CogVideoX-0.6B for T2I
- [ ] [19/1/25] Scale up to T2V on WebVid-6.6M
- [ ] [16/1/25] Implement Convolution for Wavelet Transform
- [ ] [17/1/25] Decompsed Value and Shared QK for Attention
- [ ] [17/1/25] Reimplement of NOVA without Masking Spatial Layers
