**audio-based deception detection models** using spectrogram features. The models are designed for **cross-domain generalization** and can be extended to multimodal pipelines.  

Key features:
- **ResNet50-based audio feature extractor** adapted for single-channel spectrograms.
- **Linear classifier** for binary deception prediction.
- **Custom Dataset loader** for spectrograms, including language and ethnic group filtering.
- **Training and evaluation scripts** supporting k-fold cross-validation.

## Dataset
> **Important:** The ROSE V2 audio dataset used in this work is **private and not publicly available**.  
> This repository does **not include the dataset**. Users must supply their own audio spectrogram dataset with compatible annotation files (CSV format).
