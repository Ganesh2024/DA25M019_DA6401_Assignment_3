import math
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# Google Drive file IDs for auto-download
VOCAB_GDRIVE_FILE_ID = "1ygnQcnD-GRrf8xsvzwAYKfBWGFJghlfw"
WEIGHTS_GDRIVE_FILE_ID = "16KyLen3MM35md8YkTHK6hISZaS_O1Fqu"


def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive if not exists"""
    if not os.path.exists(output_path):
        try:
            import gdown
            gdown.download(id=file_id, output=output_path, quiet=False)
        except Exception as e:
            print(f"Could not download {output_path}: {e}")


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn_w = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    batch, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
    return pad_mask | causal


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = query.size(0)
        Q = self.W_q(query).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = 1,
        tgt_vocab_size: int = 1,
        d_model: int = 256,
        N: int = 3,
        num_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 1,
        checkpoint_path: str = None,
    ) -> None:
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model

        import spacy

        vocab_file = "vocabs.pt"
        weights_file = "best_checkpoint.pt"

        download_from_gdrive(VOCAB_GDRIVE_FILE_ID, vocab_file)

        saved = torch.load(vocab_file, map_location="cpu")
        self.src_vocab = saved["src_vocab"]
        self.tgt_vocab = saved["tgt_vocab"]
        self.tgt_itos  = saved["tgt_itos"]
        src_vocab_size = len(self.src_vocab)
        tgt_vocab_size = len(self.tgt_vocab)

        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            from spacy.cli import download as spacy_download
            spacy_download("de_core_news_sm")
            self.spacy_de = spacy.load("de_core_news_sm")

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)

        self.src_embed = nn.Sequential(
            nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx),
            PositionalEncoding(d_model, dropout),
        )
        self.tgt_embed = nn.Sequential(
            nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx),
            PositionalEncoding(d_model, dropout),
        )
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.proj = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

        download_from_gdrive(WEIGHTS_GDRIVE_FILE_ID, weights_file)
        ckpt = torch.load(weights_file, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        self.load_state_dict(state, strict=False)
        print("Weights loaded successfully.")

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.proj(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        self.eval()
        device = next(self.parameters()).device

        tokens = [tok.text.lower() for tok in self.spacy_de.tokenizer(src_sentence)]
        unk_idx = self.src_vocab.get("<unk>", 0)
        sos_idx = self.src_vocab.get("<sos>", 2)
        eos_idx = self.src_vocab.get("<eos>", 3)
        tgt_sos = self.tgt_vocab.get("<sos>", 2)
        tgt_eos = self.tgt_vocab.get("<eos>", 3)
        pad_idx = self.pad_idx

        ids = [sos_idx] + [self.src_vocab.get(t, unk_idx) for t in tokens] + [eos_idx]
        src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = make_src_mask(src, pad_idx)

        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys = torch.tensor([[tgt_sos]], dtype=torch.long, device=device)
            for _ in range(100):
                tgt_mask = make_tgt_mask(ys, pad_idx)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == tgt_eos:
                    break

        out_ids = ys[0, 1:].tolist()
        special = {tgt_sos, tgt_eos, pad_idx}
        words = [self.tgt_itos[i] for i in out_ids if i not in special]
        return " ".join(words)