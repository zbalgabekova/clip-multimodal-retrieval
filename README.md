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


