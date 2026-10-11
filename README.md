# Image Captioning with CLIP & Transformers

Generates natural-language captions for images. A frozen **CLIP ViT-B/32** encodes the image and a **Transformer decoder trained from scratch** writes the caption. Trained on **MS COCO 2014** using the standard **Karpathy split** and scored with the official COCO caption toolkit, so the numbers can be compared directly with published results.

## Results (COCO Karpathy test split, 5,000 images)

| Metric | Score |
| :--- | :---: |
| BLEU-1 | 73.64 |
| BLEU-2 | 57.30 |
| BLEU-3 | 43.76 |
| BLEU-4 | 33.36 |
| METEOR | 28.15 |
| ROUGE-L | 55.92 |
| **CIDEr** | **112.67** |

Comparison with published models trained with cross-entropy loss only (no reinforcement-learning fine-tuning) on the same split:

| Model | Trainable params | BLEU-4 | METEOR | CIDEr |
| :--- | :---: | :---: | :---: | :---: |
| **This project (frozen CLIP ViT-B/32 + Transformer decoder)** | 27M | **33.4** | **28.2** | **112.7** |
| Bottom-Up Top-Down (Anderson et al., 2018) | 52M | 36.2 | 27.0 | 113.5 |
| ClipCap, Transformer + frozen GPT-2 (Mokady et al., 2021) | 43M | 33.5 | 27.5 | 113.1 |
| ClipCap, MLP + fine-tuned GPT-2 (Mokady et al., 2021) | 156M | 32.2 | 27.1 | 108.4 |

Comparison numbers are from Table 1 of the [ClipCap paper](https://arxiv.org/abs/2111.09734). This model comes within 1 CIDEr point of ClipCap and Bottom-Up Top-Down with fewer trainable parameters, without a pretrained language model or object detector. Training takes **~80 minutes on a laptop RTX 4050 (6 GB)**, plus a one-time ~6 minutes to cache the CLIP features and ~8 minutes for test evaluation.

## How it works

* **Encoder:** frozen `openai/clip-vit-base-patch32`. Each image becomes 50 tokens (1 class token + 7×7 patches). These are computed once and cached to disk as float16 (~9.5 GB), so training never runs the vision model.
* **Decoder:** a 4-layer Transformer decoder (d_model 512, 8 heads, 10k-word vocabulary), built from scratch, with causal self-attention over the caption and cross-attention over the projected CLIP tokens.
* **Training:** AdamW with linear warmup and cosine decay, label smoothing 0.1, bfloat16 autocast, and gradient clipping. The model with the best val loss is kept, and training stops early after 3 epochs without improvement. In this run, epoch 8 of 11 was best.
* **Decoding:** beam search (beam 3) with **trigram blocking**, which stops repetitive loops like *"a girl in a pink shirt and a girl in a pink shirt"*.
* **Evaluation:** `pycocoevalcap` (PTB tokenizer, BLEU, METEOR, ROUGE-L, CIDEr) on the 5,000-image test split. The val split is only used for checkpoint selection.

## Quick start

```bash
git clone https://github.com/vexyiwnl/image-captioning-project-.git
cd image-captioning-project-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install default-jre        # METEOR and the PTB tokenizer need Java
```

Download COCO into `coco/` (the commands are in the notebook's first cell), then open `coco_clip_captioning.ipynb` and run all cells. Re-running the training cell resumes from the last checkpoint.

## Earlier version: ResNet-101 on Flickr30k

`flickr30k_image_captioning.ipynb` is the first version of this project: a ResNet-101 encoder (last block fine-tuned) that outputs a 7×7 grid of 49 visual tokens, paired with the same style of Transformer decoder, trained on Flickr30k with AMP and dynamic padding. It reached a BLEU-4 of 22.81 (NLTK corpus BLEU on its validation split). Because it uses a different dataset, split and scorer, that number can't be compared with the COCO results above. Moving to COCO + CLIP and the standard evaluation toolkit was what made a comparison with published work possible.

Its checkpoint is [on Google Drive](https://drive.google.com/file/d/1hkLgYBT7KS7Hw1JuCtUrzTuFlLjQh6xO/view?usp=sharing).
