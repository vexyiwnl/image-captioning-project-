# 👁️📝 Image Captioning with ResNet-101 & Transformers

An end-to-end deep learning pipeline for generating descriptive text from images, trained on the **Flickr30k** dataset. This project implements a custom sequence-to-sequence architecture utilizing a Convolutional Neural Network (CNN) as the visual encoder and an autoregressive Transformer as the caption decoder.

## 🧠 Architecture Overview

* **Visual Encoder (ResNet-101):** Extracts a `7x7` spatial feature grid from input images, flattened into 49 distinct visual tokens.
* **Caption Decoder:** A custom PyTorch `nn.TransformerDecoderLayer` built from scratch. It uses masked self-attention for the generated text and cross-attention to ground the text generation in the extracted visual tokens.
* **Inference:** Features a custom implementation of **Beam Search** to accumulate log-probabilities and generate highly fluent captions rather than relying on greedy decoding.

## ⚙️ Hardware & Training Optimizations
To maximize throughput and stabilize training on consumer hardware (RTX 4050), several optimizations were implemented:
* **Automatic Mixed Precision (AMP):** Utilized `torch.cuda.amp.autocast` and `GradScaler` alongside `allow_tf32=True` for accelerated tensor core utilization.
* **Dynamic Sequence Padding:** Implemented a custom `collate_fn` using `pad_sequence` to avoid global padding overhead, drastically speeding up data loading.
* **Regularization:** Label Smoothing (0.1), Dropout, and Weight Decay (1e-4) to prevent overfitting on the Flickr30k vocabulary.

## 📊 Performance & Evaluation
The model was evaluated using **Corpus BLEU** against the 5 human-annotated reference captions per image in the Flickr30k validation set.

| Metric | Score | 
| :--- | :--- |
| **BLEU-1** | 64.12 |
| **BLEU-2** | 46.54 |
| **BLEU-3** | 32.74 |
| **BLEU-4** | **22.81** |

*A BLEU-4 score of 22.81 demonstrates strong spatial understanding and the ability to generate coherent, multi-word descriptive phrases.*

## 🚀 Quick Start

### Prerequisites
* Python 3.8+
* PyTorch (with CUDA support)
* Pillow, NLTK, Pandas, tqdm

### Inference
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flickr30k-image-captioning.git
   cd flickr30k-image-captioning

2. Add checkpoint:
    Download the checkpoints folder from: https://drive.google.com/file/d/1hkLgYBT7KS7Hw1JuCtUrzTuFlLjQh6xO/view?usp=sharing
    Paste it in the project folder

3. Then create a new cell and run:
    ```
    import torch
    from PIL import Image
    from torchvision import transforms

    # Setup Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Checkpoint and Vocabulary
    checkpoint = torch.load('checkpoints_flickr30k/best_model.pt', map_location=DEVICE)
    vocab = Vocabulary.from_state(checkpoint['vocab_state'])

    # Initialize Model 
    model = ImageCaptioner(
        vocab_size=len(vocab),
        d_model=512,
        n_head=8,
        n_layers=4,
        dim_ffn=2048,
        max_len=42
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Prepare Image 
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open("sample_image.jpg").convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Generate text using Beam Search
    predicted_caption = model.generate(image_tensor, vocab, max_len=42, beam_size=3)
    print("Generated Caption:", predicted_caption)
    ```