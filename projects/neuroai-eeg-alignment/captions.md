# Figure captions (draft)

**Figure A. Predicting EEG from deep-network layers.** For each layer of ImageNet-trained ResNet-50 and ViT-B/16, a ridge
regression maps activations to the EEG response to the same image (THINGS-EEG2, 9 subjects, 17 channels); scores are
Pearson r on 200 held-out test images, divided by the square root of the EEG split-half reliability (early and late panels)
or raw r maximised over time (right). Dots mark each model's best layer. Dashed lines: best layer of an untrained ResNet-50
with calibrated BatchNorm and of a non-learned Gabor-energy/Fourier-power baseline. Early layers give the sharpest early
peak; later layers give the strongest sustained (late) prediction.

**Figure B. When does each layer predict the EEG?** Noise-ceiling-normalized predictivity as a function of layer and time.
Dotted lines: stimulus onset and the onset of the next image in the 5 Hz stream.

**Figure C. Effect sizes.** Differences in normalized r between model pairs, each at its leave-one-subject-out best layer.
Error bars are 95% CIs from a two-way bootstrap over subjects and test images. Trained networks exceed untrained and
low-level baselines, most clearly after 200 ms; an untrained network is close to a Gabor/Fourier baseline; ResNet-50 and
ViT-B/16 differ little once the readout is matched.

**Figure D. Readout sensitivity.** Finer spatial pooling raises both models' scores; the ResNet-50 vs ViT-B/16 difference is
small throughout and changes sign in the early window at the finest grid. PCA size has negligible effect.
