**multimodal deception detection models**, combining **audio** and **visual (face)** cues.  

- **Audio branch:** ResNet50-based spectrogram classifier.
- **Face branch:** Slow R50 3D ResNet-based video classifier.
- **Fusion module:** Concatenation or weighted combination of audio and face embeddings for joint prediction.
- **Training pipelines:** Supports k-fold cross-validation, multi-GPU training, and flexible loss functions.

## Dataset
> **Important:** The ROSE V2 dataset used in this work is **private and not publicly available**.  
> This repository does **not include the dataset**. Users must supply their own audio spectrograms and facial video frames with compatible CSV annotations.
