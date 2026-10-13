---
title: Image Captioning (CLIP + Transformer)
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Image captioning demo

Upload an image and get a caption from a frozen CLIP ViT-B/32 encoder + 4-layer Transformer decoder
trained on COCO (Karpathy split). Test scores: BLEU-4 33.36, METEOR 28.15, CIDEr 112.67.

Training code and full results: https://github.com/vexyiwnl/image-captioning-project-

## Run locally

`caption_model.pt` is not in the GitHub repo (110 MB). Create it from the notebook's checkpoint:

```bash
python -c "import torch; c = torch.load('checkpoints_coco/best_model.pt', map_location='cpu', weights_only=True); torch.save({k: c[k] for k in ('model', 'vocab', 'model_config', 'clip_name')}, 'demo/caption_model.pt')"
pip install gradio
python demo/app.py
```
