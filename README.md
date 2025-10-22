#  Avatar Type Recognition 

## Overview
This project presents a complete experimental study on **avatar image classification** across three domains:
**real photos**, **drawn illustrations**, and **AI-generated avatars**.  
Nine convolutional architectures were trained, evaluated, and compared using a unified experimental protocol,  
focusing on **accuracy, calibration, robustness, bias, efficiency, and generalization**.

---

##  Research Goal
To design and compare deep convolutional neural network models capable of recognizing the **type of avatar image**,  
and to analyze their **explainability, domain bias, and generalization ability** under various real-world distortions.

---

##  Datasets
- **Training / Validation** — mixed dataset of ~3K avatar images categorized into:
  - `real` — real human photos,
  - `drawing` — hand-drawn or illustrated portraits,
  - `generated` — AI-synthesized avatars.
- **Test set (Block 9)** — 300 unseen mixed images (`real_test`, `drawn_test`, `AI_test`).
- **Out-of-Domain (Block 16)** — 5 external sets:
  - `children_adults` (human age variation)
  - `obj` (non-human objects)
  - `simpsons` (cartoon faces)
  - `animal_faces` (animal avatars)
  - `muffin_vs_chihuahua` (visual ambiguity test)

All images were normalized to `224×224` px, augmented (flip, rotation, noise), and loaded with PyTorch DataLoader.

---

##  Model Architectures
The study compares **9 convolutional models** trained under identical conditions:

| Architecture | Training Mode | Description |
|---------------|---------------|--------------|
| **MobileNetV3 Small 100** | Full | Lightweight CNN baseline optimized for mobile inference |
| **ResNet-50** | Full | Strong baseline with skip connections |
| **EfficientNet-B0** | Frozen Backbone | Transfer-learning baseline |
| **ConvNeXt-Tiny (Stage 1/2)** | Progressive Unfreeze | Modernized ResNet family, two-stage fine-tuning |
| **MobileNetV3 Few-Shot (4 / 12 epochs)** | Low-data | Generalization from small datasets |
| **ResNet-18 Few-Shot (4 / 12 epochs)** | Low-data | Compact residual model for small data regimes |

Each network was trained with **CrossEntropyLoss**, **Adam optimizer**,  
and validated on identical folds for comparability.

---

##  Experimental Blocks
All analysis was performed in 18 structured blocks:

| № | Block | Goal |
|---|-------|------|
| **9** | Unified blind test (300 images) | Evaluate all models on unseen mixed data |
| **10** | Robustness test | Blur, noise, rotation, JPEG, brightness distortions |
| **11** | Softmax confidence & calibration | Reliability diagrams, Expected Calibration Error (ECE) |
| **12** | Feature embedding visualization | t-SNE projection of latent features |
| **12B** | Cluster structure analysis | Silhouette, intra/inter-class distance |
| **13** | Ensemble soft-voting | Combined prediction across models |
| **13B–13C** | Ensemble correlation & weighting | Weighted voting based on accuracy × (1 – ECE) |
| **14–14C** | Grad-CAM explainability | Heatmap visualization and model × class comparison walls |
| **15** | Efficiency benchmark | FPS / latency (CPU & GPU) vs. model complexity |
| **16** | Out-of-Domain & Bias test | Generalization to unseen domains |
| **17** | Error & Bias analysis | Model bias maps, confidence distribution |
| **18** | Final integrated report | Normalized metrics & integrated score synthesis |

---

##  Key Metrics
Each model was evaluated by:
- `Accuracy@300` — main test accuracy,
- `Robustness` — mean accuracy under 5 distortions,
- `ECE` — calibration error,
- `Unbias` — fairness under out-of-domain images,
- `Silhouette / SeparationRatio` — feature space quality,
- `FPS / Params(M)` — computational efficiency,
- `Integrated Score` — normalized aggregate index.

---

##  Main Findings
- **ConvNeXt-Tiny Stage 2** achieved top overall accuracy and well-balanced calibration.  
- **ResNet18 Few-Shot 12 epochs** showed the best calibration (lowest ECE).  
- **MobileNetV3 Few-Shot 12 epochs** offered the best trade-off between speed and accuracy.  
- **Soft-Voting Ensemble** outperformed individual models by ≈ 4-5 % F1-gain.  
- **Out-of-Domain Bias** revealed overconfidence on non-human imagery — mitigated in ensemble.  
- **Grad-CAM Analysis** confirmed class-specific attention zones and interpretable focus maps.  
- **Efficiency Benchmark** showed MobileNetV3 > 100 FPS on GPU and ~15 FPS on CPU.

---

##  Visual Results

<p align="center">
  <img src="https://github.com/Figrac0/Avatar-Type-Recognition/blob/main/assets/ResNet18_FewShot12ep_reliability.png" width="450"/><br/>
  <em>Reliability diagram — ResNet18 Few-Shot 12 epochs (calibration curve)</em>
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

##  Summary
This project provides a **full-stack deep-learning benchmark** for avatar type recognition:
- from **training and evaluation** to **explainability and bias auditing**,
- fully reproducible via Colab (works on CPU or GPU),
- delivering both **scientific metrics** and **visual analytics**.

# 🧠 Распознавание типов аватаров — исследовательский проект

## Обзор
Данный проект представляет собой полный экспериментальный цикл по классификации изображений аватаров по трём доменам:  
**реальные фотографии**, **рисованные иллюстрации** и **AI-сгенерированные изображения**.  
Были обучены, протестированы и сравнены девять сверточных нейронных архитектур в единой экспериментальной среде,  
с акцентом на **точность, калибровку, устойчивость, смещение, эффективность и обобщающую способность**.

---

## Цель исследования
Разработать и сравнить сверточные нейронные сети, способные определять **тип изображения аватара**,  
а также проанализировать их **интерпретируемость, склонность к смещению и способность к обобщению**  
на данных, отличных от обучающего домена.

---

## Датасеты
- **Обучающая и валидационная выборки** — около 3 000 изображений, разделённых на категории:
  - `real` — реальные фотографии людей;
  - `drawing` — рисованные или иллюстрированные портреты;
  - `generated` — изображения, созданные с помощью нейросетей.
- **Тестовая выборка (Блок 9)** — 300 ранее не встречавшихся изображений (`real_test`, `drawn_test`, `AI_test`).
- **Вне-доменные данные (Блок 16)** — 5 отдельных наборов:
  - `children_adults` — детские и взрослые лица;
  - `obj` — фотографии объектов и фруктов;
  - `simpsons` — персонажи мультфильма «Симпсоны»;
  - `animal_faces` — лица животных;
  - `muffin_vs_chihuahua` — неоднозначные изображения.

Все данные были нормализованы до размера `224×224`, аугментированы (повороты, шум, зеркалирование)  
и загружались через `PyTorch DataLoader`.

---

## Архитектуры моделей
В исследовании сравнивались **9 моделей CNN**, обученных в одинаковых условиях:

| Архитектура | Режим обучения | Описание |
|--------------|----------------|-----------|
| **MobileNetV3 Small 100** | Полное обучение | Легковесная архитектура, оптимизированная для мобильных устройств |
| **ResNet-50** | Полное обучение | Базовая мощная модель с остаточными связями |
| **EfficientNet-B0** | Замороженный бэкбон | Трансферное обучение с фиксированными весами базовых слоёв |
| **ConvNeXt-Tiny (Stage 1/2)** | Прогрессивное размораживание | Современная версия ResNet с двухэтапной дообучаемостью |
| **MobileNetV3 Few-Shot (4 / 12 эпох)** | Малые данные | Обучение на ограниченной выборке для проверки обобщающей способности |
| **ResNet-18 Few-Shot (4 / 12 эпох)** | Малые данные | Компактная остаточная сеть для быстрого обучения |

Все модели обучались с использованием **CrossEntropyLoss**, оптимизатора **Adam**  
и проверялись на одинаковых наборах данных для честного сравнения.

---

## Экспериментальные блоки
Работа состоит из 18 исследовательских блоков:

| № | Блок | Назначение |
|---|------|-------------|
| **9** | Слепое тестирование (300 изображений) | Проверка моделей на неизвестных данных |
| **10** | Тест на устойчивость | Шум, размытие, яркость, поворот, JPEG-сжатие |
| **11** | Анализ уверенности Softmax | Диаграммы надёжности, ошибка калибровки (ECE) |
| **12** | Визуализация признаков | Проекции t-SNE внутренних представлений |
| **12B** | Анализ кластеров | Метрики Silhouette и меж-/внутриклассовые расстояния |
| **13** | Энсемблирование (Soft-Voting) | Объединение предсказаний нескольких моделей |
| **13B–13C** | Корреляция и взвешивание | Soft-Voting с весами по точности и калибровке |
| **14–14C** | Grad-CAM интерпретация | Тепловые карты внимания и сравнительные коллажи |
| **15** | Бенчмарк эффективности | Измерение FPS, времени инференса и сложности моделей |
| **16** | Вне-доменный тест | Проверка обобщающей способности на нестандартных данных |
| **17** | Анализ ошибок и смещений | Карты смещений и распределение уверенности |
| **18** | Финальный отчёт | Свод нормализованных метрик и интегрального скора |

---

## Ключевые метрики
Каждая модель оценивалась по следующим показателям:
- **Accuracy@300** — точность на тестовой выборке;
- **Robustness** — средняя точность при искажениях;
- **ECE** — ошибка калибровки;
- **Unbias** — устойчивость к смещению на вне-доменных данных;
- **Silhouette / SeparationRatio** — качество разделения в пространстве признаков;
- **FPS / Params(M)** — вычислительная эффективность;
- **Integrated Score** — интегральная нормализованная оценка.

---

## Основные результаты
- **ConvNeXt-Tiny Stage 2** показала наилучший баланс точности и калибровки.  
- **ResNet-18 Few-Shot 12 эпох** имела минимальный ECE (лучшая калибровка).  
- **MobileNetV3 Few-Shot 12 эпох** обеспечила лучший компромисс между скоростью и качеством.  
- **Soft-Voting Ensemble** улучшил F1-метрику на ≈ 4–5 %.  
- **Out-of-Domain Test** выявил избыточную уверенность некоторых моделей на не-человеческих изображениях,  
  что было сглажено при использовании ансамбля.  
- **Grad-CAM** визуализировал корректные зоны внимания и интерпретируемость моделей.  
- **Бенчмарк эффективности** показал, что MobileNetV3 достигает > 100 FPS на GPU и ≈ 15 FPS на CPU.

---

## Итог
Проект представляет собой **полноценный стек экспериментов** по распознаванию типов аватаров:
- от **обучения и тестирования** до **интерпретации и аудита смещений**;
- полностью воспроизводим в **Google Colab** (CPU / GPU);
- сочетает **научную строгость** и **визуальную аналитичность**.
