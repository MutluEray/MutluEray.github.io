# Decoding figure captions (draft)

**Figure E. Identifying the viewed image from EEG with the encoding models.** Each layer's encoding model predicts the EEG
response to all 200 held-out images; each observed response is matched to the most similar prediction (200-way, chance
0.5%). Solid lines: 80 test repetitions averaged; dotted: 10. Dashed: untrained ResNet-50 and Gabor/Fourier baselines.

**Figure F. Decoding EEG into network feature space.** (F1) A ridge model maps EEG to each layer's leading feature
components; the viewed image is retrieved among the 200 test images by feature similarity. (F2) Randomly chosen examples;
green frames mark the exact image.

**Figure G. What can be linearly recovered about the image?** 16x16 RGB reconstructions from EEG in four equal-length
(100 ms) time windows (upper rows) and from network layers (remaining rows). Networks show what each layer linearly carries
(an upper reference); EEG recovers only coarse luminance and colour layout. Temporal structure is analysed in the encoding
results (layer x time maps), not here.
