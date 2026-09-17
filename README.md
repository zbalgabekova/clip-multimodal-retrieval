# Clip Multimodal Retrieval

A multimodal computer vision project using **CLIP (Contrastive Language–Image Pre-training)** for cross-modal image-text retrieval.
The project fine-tunes a pretrained **CLIP ViT-B/32** model on the **Flickr30K** dataset and evaluates its ability to retrieve relevant images from text queries and relevant captions from images.

The project includes baseline evaluation, contrastive fine-tuning, retrieval metrics, qualitative analysis, and an interactive Gradio demo.


## 🚀 Live Demo

Try the fine-tuned model interactively:

**Hugging Face Spaces:**  
[clip-flickr30k-retrieval](https://huggingface.co/spaces/zaremabalgabekova/clip-flickr30k-retrieval)

The demo supports:

- **Text → Image retrieval**
- **Image → Text retrieval**

The retrieval database contains a subset of the Flickr30K test set included with the demo.

---

## 📌 Project Overview

CLIP learns a shared embedding space for images and natural-language descriptions.

In this project, the model is used for two cross-modal retrieval tasks:

### Text → Image

Given a natural-language query:

> "A child playing outside near a wooden house"

the model retrieves images whose visual content is most similar to the text.

### Image → Text

Given an image, the model retrieves the captions that are most semantically similar to the image.

The project compares the performance of:

1. Pretrained CLIP
2. Fine-tuned CLIP on Flickr30K

---

## 📊 Dataset

The project uses the **Flickr30K** image-caption dataset.

The original dataset contains:

- 31,783 images
- 158,915 captions
- 5 captions per image

### Dataset Split

The dataset was split by image, ensuring that images do not appear across multiple splits.

| Split      |     Images |    Captions |
| ---------- | ---------: | ----------: |
| Train      |     25,426 |     127,130 |
| Validation |      3,178 |      15,890 |
| Test       |      3,179 |      15,895 |
| **Total**  | **31,783** | **158,915** |

---

## 🤖 Model

The project uses:

**CLIP ViT-B/32**

Base model:

'openai/clip-vit-base-patch32'

The pretrained CLIP model contains separate image and text encoders that project both modalities into a shared embedding space.

For retrieval, embeddings are L2-normalized and compared using cosine similarity.

---

## 🔬 Fine-Tuning

The model is fine-tuned using a symmetric contrastive loss.

A subset of 5,000 training images was used for fine-tuning due to computational constraints.

---

## 📈 Results

### Text → Image Retrieval

| Model               |        R@1 |        R@5 |       R@10 |
| ------------------- | ---------: | ---------: | ---------: |
| Pretrained CLIP     |     44.70% |     71.08% |     79.62% |
| **Fine-tuned CLIP** | **51.58%** | **77.22%** | **85.25%** |
| Improvement         |   +6.88 pp |   +6.14 pp |   +5.63 pp |

### Image → Text Retrieval

| Model               |        R@1 |        R@5 |       R@10 |
| ------------------- | ---------: | ---------: | ---------: |
| Pretrained CLIP     |     67.91% |     89.15% |     94.18% |
| **Fine-tuned CLIP** | **70.37%** | **90.72%** | **95.06%** |
| Improvement         |   +2.46 pp |   +1.57 pp |   +0.88 pp |


## 🔎 Qualitative Retrieval

In addition to numerical metrics, retrieved results are inspected visually.

**Text → Image retrieval**

### Successful case

<p align="center">
  <img src="images/text-to-image/perfect/caption_0.png" width="900">
</p>

### Partially correct case

<p align="center">
  <img src="images/text-to-image/hard/caption_1.png" width="900">
</p>

**Image → Text Retrieval**

### Successful case

<p align="center">
  <img src="images/image-to-text/perfect/image_4.png" width="900">
</p>

### Partially correct case

<p align="center">
  <img src="images/image-to-text/partial/image_0.png" width="900">
</p>

Qualitative evaluation helps identify cases where the model retrieves semantically related images even when they do not exactly match the Flickr30K ground-truth annotation.

---




