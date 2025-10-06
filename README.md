# deception-detection-dd

This repository contains research-grade implementations of **multimodal deception detection** models, focusing on **audio** and **video/face** modalities. The code is designed to be modular, extensible, and suitable for cross-domain, cross-ethnic deception detection research. 

This is part of my **Master’s thesis** work at Nanyang Technological University **(NTU)**, Singapore, conducted under the MSc in Communication Engineering program.

🎯 Research Objective

The primary objective of this research is to develop a fair and robust multimodal deception detection system that generalizes across speakers of different ethnicities and languages.


**Key Features:**
- **Audio-based DD:** Uses spectrogram inputs with a ResNet50 backbone.
- **Video/Face-based DD:** Uses 3D CNN (Slow_R50) to process video sequences.
- **Clip-level aggregation:** Handles long video sequences by splitting and averaging predictions.
- **K-fold cross-validation:** Built-in support for evaluation across multiple folds.
- **GPU & multi-GPU training:** Optimized for high-performance training.
- **Modular architecture:** Easy to extend or integrate with additional modalities or fusion methods.

## Dataset
> **Important:** Because the project is still in progress, the **ROSE v2** dataset is not yet public. 
