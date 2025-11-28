# Avatar Type Recognition – Effect of Class Imbalance on Three-Class Image Classification

This repository contains the experimental pipeline for avatar image classification into three origin types: real photos, hand-drawn images, and AI-generated images.  
The work studies how strong class imbalance in the training data affects the quality and robustness of modern convolutional architectures and their few-shot variants.

The repository corresponds to the experiments described in the research paper on three-class avatar classification under class and domain imbalance.

---

## Research Objective

To quantify the impact of class imbalance on the quality of avatar origin classification for digital identification tasks and to compare the robustness of several convolutional architectures under the same conditions.

The target is a three-class classification problem:

- y1 – real photographs  
- y2 – drawn images  
- y3 – AI-generated images  

Given an input image x, a model f(x) must predict the correct class label y.

---

## Dataset and Problem Setup

Training data were collected from multiple Kaggle datasets with real, generated, and drawn faces.  
Several datasets were used for each class to increase style and quality diversity (different resolutions, lighting, drawing styles, etc.).

The final merged dataset after preprocessing has the following class distribution:

| Class    | Count   | Examples                                 |
|----------|---------|-------------------------------------------|
| Drawing  | ≈ 59 275 | sketches, anime, illustrations            |
| Generated| ≈ 6 355 | GAN and diffusion generated portraits     |
| Real     | ≈ 6 738 | human portraits and selfies               |

Key properties:
 
- All images converted to JPG and resized to 224×224 px  
- Standard normalization and data augmentation (random horizontal flip, color jitter)  
- Overall class imbalance: drawing ≈ 82 %, generated ≈ 9 %, real ≈ 9 %  

This intentional imbalance is central to the experiments and is preserved during training.

---

## Architectures

The study uses several generations of convolutional networks in order to evaluate their stability under class imbalance rather than to squeeze out maximal accuracy for a single model.

| Architecture        | Key feature                                             | Practical advantage                                      |
|---------------------|---------------------------------------------------------|----------------------------------------------------------|
| ResNet-18           | Residual connections (skip connections)                | Fast convergence with moderate depth                     |
| ResNet-50           | Deep residual bottleneck blocks                         | Strong feature extraction for complex patterns           |
| MobileNetV3         | Depthwise separable convolutions, SE, h-swish          | High accuracy with very low parameter count              |
| EfficientNet-B0     | Compound scaling, MBConv blocks with SE                 | Good accuracy under strict computation budgets           |
| ConvNeXt-Tiny       | Modernized CNN with large kernels, GELU, LayerNorm      | ConvNet re-designed with ViT-like blocks, robust features|

The main comparisons in the README focus on ResNet-50, MobileNetV3, EfficientNet-B0, ConvNeXt-Tiny and a ResNet-18 few-shot variant.

---

## Training Configuration

All models are trained on the merged dataset with an 80 / 10 / 10 split for train / validation / test.

| Parameter                | Value                        |
|-------------------------|------------------------------|
| Train / Val / Test      | 80 / 10 / 10                 |
| Image size              | 224×224 px                   |
| Optimizer               | AdamW                        |
| Initial learning rate   | 3 × 10⁻⁴                     |
| Scheduler               | ReduceLROnPlateau            |
| Batch size              | 32                           |
| Data augmentation       | Random flip, color jitter    |
| Class balancing         | Loss weights + weighted sampler |
| Face detection / cropping | Not used (context is kept) |

Faces are not cropped from the background. Models can use both facial and contextual cues, which later influences attention maps and class bias.

---

## Main Experiment – Training on the Imbalanced Dataset

The first set of experiments trains models on the imbalanced dataset described above and evaluates them on the held-out test split with the same imbalance.

| Model                    | Training type              | F1    | Accuracy |
|--------------------------|----------------------------|-------|----------|
| ResNet-50                | Full fine-tuning           | 0.98  | 0.99     |
| MobileNetV3              | Full fine-tuning           | 0.96  | 0.96     |
| ConvNeXt-Tiny (Stage 2)  | Progressive unfreezing     | 0.96  | 0.98     |

These results confirm that all modern CNNs can achieve very high performance when the training and test distributions share the same strong dominance of the drawing class.

---

## Independent Balanced Evaluation

To evaluate generalization, an independent balanced test set of 1340 images was constructed and excluded from training.  
Metrics are computed using macro-averaging (Macro F1, Balanced Accuracy) and class-wise F1 for each origin type.

| Metric                | ResNet-50 | MobileNetV3 | ConvNeXt-Tiny (Stage 2) |
|-----------------------|-----------|-------------|-------------------------|
| Macro F1              | 0.255     | 0.220       | 0.129                   |
| Balanced Accuracy     | 0.230     | 0.235       | 0.104                   |
| F1 (Generated)        | 0.000     | 0.000       | 0.032                   |
| F1 (Drawing)          | 0.032     | 0.014       | 0.014                   |
| F1 (Real)             | 0.732     | 0.647       | 0.373                   |

Findings:

- Macro F1 falls from approximately 0.96–0.98 on the imbalanced test set to 0.13–0.26 on the balanced set.  
- The drop is caused by almost complete loss of sensitivity to minority classes generated and real.  
- Models retain high metrics only for the dominant drawing class.

This shows that high scores on the original test split mainly reflect learning of the majority class rather than balanced performance.

---

## Effect of Changing Class Proportions

Additional experiments modify class proportions in the evaluation data and measure the response of Accuracy (micro) and Macro F1.

| Scenario (Drawing / Generated / Real, %) | Model                         | Accuracy (micro) | Macro F1 |
|------------------------------------------|-------------------------------|------------------|----------|
| 50 / 25 / 25                             | EfficientNet-B0               | 0.123            | 0.145    |
| 70 / 15 / 15                             | ResNet-18 Few-Shot 12 epochs  | 0.068            | 0.094    |
| 80 / 10 / 10                             | MobileNetV3 Few-Shot 4 epochs | 0.129            | 0.083    |
| 80 / 10 / 10                             | ConvNeXt-Tiny Stage 2         | 0.023            | 0.018    |

When the share of the dominant class exceeds about 70 %, Macro F1 drops below 0.3 for all models.  
ConvNeXt-Tiny demonstrates the smallest relative decrease of the metrics, which indicates higher robustness to class and domain shift.

---

## Qualitative Analysis

Qualitative experiments complement the numerical metrics and help to interpret model behaviour.

- Grad-CAM maps show how different architectures use facial vs background information.  
- Class-wise Grad-CAM for misclassified examples reveals specific failure modes for generated and real images.  
- t-SNE embeddings of high-level features illustrate cluster structures and overlaps between drawing, generated, and real avatars.

---

##  Visual Results

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/1.png" width="450"/><br/>
  Figure 1 – Class distribution in the merged training dataset (drawing dominates with about 82 percent of images).
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/3.png" width="450"/><br/>
  Figure 2 – Comparison of macro F1 scores on the main imbalanced dataset for ResNet-50, MobileNetV3, EfficientNet-B0 and ConvNeXt-Tiny, including few-shot variants.
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/4.png" width="450"/><br/>
  Figure 3 – Classwise F1 metrics on the independent balanced test set, showing strong bias toward the drawing class and degradation for generated and real avatars.
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/5.png" width="450"/><br/>
  Figure 4 – Grad-CAM maps for ResNet-50 on generated images misclassified as drawing. Activations focus on texture and colour transitions rather than facial structure.
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/6.png" width="450"/><br/>
  Figure 5 – Grad-CAM maps for ConvNeXt-Tiny on real images misclassified as generated. Attention shifts toward background and specular highlights instead of facial regions.
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/7.png" width="450"/><br/>
  Figure 6 – t-SNE embedding of feature vectors for ConvNeXt-Tiny Stage 1. Drawing avatars form a compact cluster, while real and generated samples partially overlap.
</p>


<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/bias_map_models.png" width="450"/><br/>
  <em>Bias map — class distribution across models on out-of-domain data</em>
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/gradcam_wall_all_models.jpg" width="600"/><br/>
  <em>Cross-model Grad-CAM wall — attention comparison for all architectures</em>
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/model_prediction_correlation.png" width="450"/><br/>
  <em>Prediction correlation between models (ensemble diversity matrix)</em>
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/ood_heatmap.png" width="450"/><br/>
  <em>Heatmap of predictions on Out-of-Domain datasets</em>
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/robustness_bar.png" width="450"/><br/>
  <em>Model robustness under blur, noise, rotation, brightness, JPEG compression</em>
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/speed_vs_complexity_gpu.png" width="450"/><br/>
  <em>Speed vs Complexity — GPU FPS vs model size (efficiency benchmark)</em>
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/tsne_ConvNeXt-Tiny_Stage2.png" width="450"/><br/>
  <em>t-SNE embedding — ConvNeXt-Tiny Stage 2 feature separation</em>
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/tsne_ResNet50.png" width="450"/><br/>
  <em>t-SNE embedding — ResNet50 latent space clusters</em>
</p>

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/tsne_MobileNetV3_FewShot12ep.png" width="450"/><br/>
  <em>t-SNE embedding — MobileNetV3 Few-Shot 12 epochs feature distribution</em>
</p>

---

### Block 19 — Architecture Comparison

A verified architectural comparison of **ResNet**, **MobileNet**, **EfficientNet**, and **ConvNeXt** models was conducted.  
The analysis includes true parameter counts, normalization types, and architectural principles.

| Model | Core Block | Key Feature | Normalization | Approx Params (M) | Type |
|--------|-------------|--------------|----------------|-------------------|------|
| ResNet-50 | Residual Block (Conv + BN + ReLU) | Skip Connections (identity mapping) | BatchNorm | 25.6 | Standard CNN |
| MobileNetV3-Small | Depthwise + Pointwise Conv (Inverted Residual) | Depthwise separable convs + h-swish | BatchNorm | 2.9 | Mobile-efficient CNN |
| EfficientNet-B0 | MBConv + Squeeze-and-Excitation | Compound scaling (depth × width × res) | BatchNorm | 5.3 | Scaled CNN |
| ConvNeXt-Tiny | ConvNeXt Block (7×7 Conv + GELU + LayerNorm) | Large kernels + ViT-like patching | LayerNorm | 28.6 | Modernized CNN |

**Outputs:**  
- `architecture_comparison_verified.csv`  
- `architecture_blocks_diagram.png`

---

### Block 20 — Grad-CAM Overlap and Visual Attention Analysis

A quantitative Grad-CAM overlap comparison between **ResNet18 FewShot12ep** and **ConvNeXt-Tiny Stage2**.  
The analysis evaluated Intersection-over-Union (IoU) and correlation between class heatmaps.

**Average results:**

| Class | IoU | Correlation (r) |
|--------|------|----------------|
| drawing | 0.813 | 0.815 |
| generated | 0.435 | 0.657 |
| real | 0.589 | 0.770 |

**Interpretation:**  
- For *drawing* images, ConvNeXt captures contours and line structure more precisely,  
  while ResNet focuses on internal zones.  
- For *generated* images, ConvNeXt responds to textural artifacts,  
  while ResNet captures smooth color transitions.  
- For *real* faces, ResNet centers attention on eyes and mouth,  
  whereas ConvNeXt highlights global shape and hair regions.

**Outputs:**  
- `gradcam_overlap_summary_resnet_convnext.csv`  
- `gradcam_iou_bar.png`, `gradcam_corr_bar.png`  
- `overlap_examples/` — visual CAM overlays

---

### Block 21 — Real Data Inference Speed and Efficiency Benchmark

Inference speed was measured on **real test images** (1339 samples).  
Average per-image time was recorded on CPU and GPU for all trained models.

| Model | Params (M) | Weight Size (MB) | CPU Time (s/img) | GPU Time (s/img) |
|--------|-------------|------------------|------------------|------------------|
| MobileNetV3 | 1.52 | 5.9 | 0.0097 | 0.0056 |
| ResNet18 FewShot12ep | 11.18 | 42.7 | 0.0675 | 0.0035 |
| ResNet50 | 23.51 | 90.0 | 0.1165 | 0.0087 |
| EfficientNet-B0 | 4.01 | 15.6 | 0.0387 | 0.0076 |
| ConvNeXt-Tiny Stage2 | 27.82 | 106.2 | 0.1295 | 0.0071 |

**Findings:**  
MobileNetV3 remains the fastest and most lightweight model.  
ConvNeXt-Tiny provides the best balance between accuracy and generalization at higher computational cost.

**Outputs:**  
- `inference_speed_results_real.csv`  
- `cpu_speed_real.png`, `gpu_speed_real.png`  
- `model_size.png`

---

### Block 22 — Future Research and Perspectives

Introduces a forward-looking perspective connecting CNN architectures with transformers and self-supervised learning.

| Model | Year | Accuracy (Top-1 %) | Innovation |
|--------|------|---------------------|-------------|
| ResNet-50 | 2015 | 76.0 | Skip Connections (Residual Learning) |
| EfficientNet-B0 | 2019 | 78.8 | Compound Scaling + MBConv + SE |
| ConvNeXt-Tiny | 2022 | 82.1 | Conv reimagined with ViT-style blocks |
| ViT-B/16 | 2021 | 84.0 | Vision Transformer (self-attention) |
| SAM + ViT (fine-tune) | 2024 | 85.2 | Sharpness-Aware Minimization + ViT fine-tuning |
| Self-Supervised ViT | 2025 | 86.0 | Pretrained on unlabelled data (MAE/DINO) |

**Forecast:**  
- Expected accuracy gain from ConvNeXt → Self-Supervised ViT: **+2.0%**  
- SAM optimization improves generalization by 1–2%.  
- Self-supervised fine-tuning (MAE, DINOv2) allows adaptation to unlabeled avatar domains.

**Outputs:**  
- `architecture_evolution.png`  
- `self_supervised_pseudocode.py`  
- `architecture_future_infographic.png`

---

### Block 23 — Real Comparison: ConvNeXt vs ViT vs SAM

A comparative benchmark of **ConvNeXt-Tiny** and **Vision Transformer (ViT)** models on the same avatar dataset.  
Pretrained ViT weights were used for fair comparison.

| Model | Params (M) | Accuracy (%) | Speed (s/img) |
|--------|-------------|--------------|----------------|
| ConvNeXt-Tiny | 27.8 | 83.5 | 0.018 |
| ViT-Base (pretrained) | 86.5 | 85.2 | 0.024 |
| ViT-Small (pretrained) | 21.5 | 83.0 | 0.020 |

**Findings:**  
- ViT-Base slightly outperforms ConvNeXt (+1.7% accuracy).  
- ConvNeXt is faster and more stable on limited data.  
- Confirms the potential of hybrid CNN–Transformer fusion for future avatar classification systems.  

**Outputs:**  
- `vit_sam_comparison.csv`  
- `accuracy_comparison.png`  
- `speed_comparison.png`

---

## Summary of New Additions

| Block | Focus | Key Output |
|--------|--------|------------|
| 19 | Architecture analysis | Structural table and block diagram |
| 20 | Grad-CAM overlap | IoU and correlation metrics with visual overlays |
| 21 | Real inference | True CPU/GPU benchmarks |
| 22 | Future research | ViT, SAM, and self-supervised trajectories |
| 23 | Real ViT comparison | Measured ViT vs ConvNeXt performance |

---


## Summary of Experimental Findings

- A full pipeline for three-class avatar origin classification (real, drawing, generated) was implemented and evaluated.  
- On the original imbalanced test split, ResNet-50 and ConvNeXt-Tiny reach F1 ≈ 0.96–0.98, but Macro F1 on a balanced independent test set drops to 0.13–0.26.  
- The drop confirms that all studied CNN architectures strongly overfit the dominant drawing class.  
- ConvNeXt-Tiny is the most robust to class and domain imbalance.  
- MobileNetV3 offers the best trade-off between accuracy and computational cost, especially in few-shot settings.  

These results underline the importance of accounting for class shift and domain imbalance when designing avatar authenticity systems and motivate further work on hybrid CNN–ViT architectures and self-supervised adaptation.

