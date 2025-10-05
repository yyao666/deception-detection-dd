**video-based deception detection models** using facial frames. It focuses on **temporal modeling** and **cross-domain robustness**.  

Key features:
- **Slow R50 3D ResNet** pretrained on video data for feature extraction.
- **Linear classifier** for binary deception prediction.
- **Parsed dataset loader** splitting videos into fixed time windows with adjustable frame rates.
- **Clip-level label estimation** for more robust evaluation.
- **Training and evaluation scripts** supporting k-fold cross-validation.

## Dataset
> **Important:** The ROSE V2 facial video dataset used in this work is **private and not publicly available**.  
> This repository does **not include the dataset**. Users must provide their own preprocessed face frame datasets with compatible CSV annotations.
