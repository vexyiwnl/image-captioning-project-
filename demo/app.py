"""Gradio demo: upload an image, get a caption from the COCO-trained CLIP + Transformer model.

Vocabulary and CaptionModel are copied from coco_clip_captioning.ipynb (inference parts only),
so the checkpoint's state_dict keys line up exactly.
"""
from pathlib import Path

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModel

HERE = Path(__file__).parent
BEAM_SIZE = 3
MAX_WORDS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32


class Vocabulary:
    PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3

    def __init__(self, itos):
        self.itos = list(itos)

    def decode(self, ids) -> str:
        words = []
        for idx in ids:
            if idx == self.EOS_IDX:
                break
            if idx not in (self.PAD_IDX, self.SOS_IDX):
                words.append(self.itos[idx])
        return " ".join(words)


class CaptionModel(nn.Module):
    def __init__(self, vocab_size, feature_dim, max_len, d_model, n_head, n_layers, dim_ffn, dropout):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feature_dim, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=Vocabulary.PAD_IDX)
        self.pos = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerDecoderLayer(d_model, n_head, dim_ffn, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers, norm=nn.LayerNorm(d_model))
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features.float())

    def decode(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        x = self.dropout(self.embed(tokens) + self.pos(torch.arange(length, device=tokens.device)))
        causal_mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1)
        x = self.decoder(x, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tokens.eq(Vocabulary.PAD_IDX))
        return self.fc_out(x)

    @staticmethod
    def _repeated_ngram_tokens(sequence, n: int = 3) -> set:
        # Tokens that would complete an n-gram already present in the sequence.
        prefix = sequence[len(sequence) - n + 1:]
        return {
            sequence[i + n - 1]
            for i in range(len(sequence) - n + 1)
            if sequence[i:i + n - 1] == prefix
        }

    @torch.no_grad()
    def generate(self, features: torch.Tensor, vocab: Vocabulary, beam_size: int = 3, max_words: int = MAX_WORDS) -> str:
        """Beam search for one image. `features` is (50, feature_dim). All live beams run as one batch."""
        memory = self.encode(features.unsqueeze(0))
        beams, completed = [(0.0, [Vocabulary.SOS_IDX])], []

        for _ in range(max_words + 1):
            tokens = torch.tensor([sequence for _, sequence in beams], device=memory.device)
            logits = self.decode(tokens, memory.expand(len(beams), -1, -1))[:, -1]
            log_probs = F.log_softmax(logits.float(), dim=-1)

            candidates = []
            for row, (score, sequence) in enumerate(beams):
                log_probs[row, list(self._repeated_ngram_tokens(sequence))] = float("-inf")
                values, indices = log_probs[row].topk(beam_size)
                candidates += [(score + value, sequence + [index]) for value, index in zip(values.tolist(), indices.tolist())]

            candidates.sort(key=lambda item: item[0], reverse=True)
            beams = []
            for score, sequence in candidates[:beam_size]:
                (completed if sequence[-1] == Vocabulary.EOS_IDX else beams).append((score, sequence))
            if not beams:
                break

        completed += beams
        _, best_sequence = max(completed, key=lambda item: item[0] / len(item[1]))
        return vocab.decode(best_sequence[1:])


checkpoint = torch.load(HERE / "caption_model.pt", map_location=DEVICE, weights_only=True)
vocab = Vocabulary(checkpoint["vocab"])
model = CaptionModel(**checkpoint["model_config"]).to(DEVICE).eval()
model.load_state_dict(checkpoint["model"])
clip_processor = CLIPImageProcessor.from_pretrained(checkpoint["clip_name"])
clip_model = CLIPVisionModel.from_pretrained(checkpoint["clip_name"], dtype=CLIP_DTYPE).to(DEVICE).eval()


@torch.no_grad()
def caption_image(image) -> str:
    if image is None:
        raise gr.Error("Upload an image first.")
    pixels = clip_processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
    features = clip_model(pixel_values=pixels.to(DEVICE, CLIP_DTYPE)).last_hidden_state[0]
    return model.generate(features, vocab, BEAM_SIZE)


demo = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil", label="Image"),
    outputs=gr.Textbox(label="Caption"),
    examples=[str(HERE / "example.jpg")],
    title="Image Captioning: CLIP + Transformer",
    description=(
        "Frozen CLIP ViT-B/32 encoder + 4-layer Transformer decoder trained on COCO (Karpathy split). "
        "Test set: BLEU-4 33.4, CIDEr 112.7. "
        "[Code on GitHub](https://github.com/vexyiwnl/image-captioning-project-)"
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
