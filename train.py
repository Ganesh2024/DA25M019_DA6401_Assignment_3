import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import evaluate

from model import Transformer, make_src_mask, make_tgt_mask


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = torch.log_softmax(logits, dim=-1)
        smooth_val = self.smoothing / (self.vocab_size - 2)
        with torch.no_grad():
            dist = torch.full_like(log_prob, smooth_val)
            dist.scatter_(1, target.unsqueeze(1), self.confidence)
            dist[:, self.pad_idx] = 0.0
            mask = (target == self.pad_idx)
            dist[mask] = 0.0
        loss = -(dist * log_prob).sum(dim=-1)
        return loss[~mask].mean()


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    model.train() if is_train else model.eval()
    total_loss, n_batches = 0.0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for src, tgt in data_iter:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = make_src_mask(src)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            tgt_mask = make_tgt_mask(tgt_in)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            B, T, V = logits.shape
            loss = loss_fn(logits.reshape(B * T, V), tgt_out.reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break
    return ys


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    model.eval()
    bleu_metric = evaluate.load("bleu")

    if hasattr(tgt_vocab, "get_itos"):
        itos = tgt_vocab.get_itos()
        stoi = tgt_vocab.get_stoi()
    else:
        itos = tgt_vocab.itos
        stoi = tgt_vocab.stoi

    sos_idx = stoi.get("<sos>", 2)
    eos_idx = stoi.get("<eos>", 3)
    pad_idx = stoi.get("<pad>", 1)
    special = {sos_idx, eos_idx, pad_idx}

    predictions, references = [], []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            src = src.to(device)
            src_mask = make_src_mask(src)
            for i in range(src.size(0)):
                s = src[i].unsqueeze(0)
                sm = src_mask[i].unsqueeze(0)
                out = greedy_decode(model, s, sm, max_len, sos_idx, eos_idx, device)
                pred_ids = out[0, 1:].tolist()
                pred_words = [itos[idx] for idx in pred_ids if idx not in special]

                ref_ids = tgt[i, 1:].tolist()
                ref_words = [itos[idx] for idx in ref_ids if idx not in special]

                predictions.append(" ".join(pred_words))
                references.append([" ".join(ref_words)])

    result = bleu_metric.compute(predictions=predictions, references=references)
    return result["bleu"] * 100


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    cfg = {
        "src_vocab_size": model.proj.out_features,
        "tgt_vocab_size": model.proj.out_features,
        "d_model": model.d_model,
        "N": len(model.encoder.layers),
        "num_heads": model.encoder.layers[0].self_attn.num_heads,
        "d_ff": model.encoder.layers[0].ff.linear1.out_features,
        "dropout": model.encoder.layers[0].dropout.p,
    }
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": cfg,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt.get("epoch", 0)


class TokenDataset(Dataset):
    def __init__(self, pairs, pad_idx=1):
        self.pairs = pairs
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch, pad_idx=1):
    srcs, tgts = zip(*batch)
    src_max = max(len(s) for s in srcs)
    tgt_max = max(len(t) for t in tgts)
    src_padded = torch.tensor([s + [pad_idx] * (src_max - len(s)) for s in srcs], dtype=torch.long)
    tgt_padded = torch.tensor([t + [pad_idx] * (tgt_max - len(t)) for t in tgts], dtype=torch.long)
    return src_padded, tgt_padded


def run_training_experiment() -> None:
    import wandb
    import os
    import torch
    from dataset import Multi30kDataset
    from lr_scheduler import NoamScheduler
    from functools import partial

    CFG = dict(
        d_model=256,
        N=3,
        num_heads=8,
        d_ff=512,
        dropout=0.1,
        warmup_steps=4000,
        num_epochs=18,
        batch_size=128,
        label_smoothing=0.1,
    )

    wandb.init(project="da6401-a3", config=CFG)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = Multi30kDataset("train")
    val_ds = Multi30kDataset("validation")
    test_ds = Multi30kDataset("test")

    train_ds.build_vocab(train_ds.data, min_freq=2)
    val_ds.src_vocab = test_ds.src_vocab = train_ds.src_vocab
    val_ds.tgt_vocab = test_ds.tgt_vocab = train_ds.tgt_vocab
    val_ds.src_itos = test_ds.src_itos = train_ds.src_itos
    val_ds.tgt_itos = test_ds.tgt_itos = train_ds.tgt_itos

    train_pairs = train_ds.process_data()
    val_pairs = val_ds.process_data()
    test_pairs = test_ds.process_data()

    pad_idx = train_ds.tgt_vocab["<pad>"]

    _collate = partial(collate_fn, pad_idx=pad_idx)
    train_loader = DataLoader(TokenDataset(train_pairs), batch_size=CFG["batch_size"], shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(TokenDataset(val_pairs), batch_size=CFG["batch_size"], shuffle=False, collate_fn=_collate)
    test_loader = DataLoader(TokenDataset(test_pairs), batch_size=64, shuffle=False, collate_fn=_collate)

    model = Transformer(
        src_vocab_size=len(train_ds.src_vocab),
        tgt_vocab_size=len(train_ds.tgt_vocab),
        d_model=CFG["d_model"],
        N=CFG["N"],
        num_heads=CFG["num_heads"],
        d_ff=CFG["d_ff"],
        dropout=CFG["dropout"],
        pad_idx=pad_idx,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=CFG["d_model"], warmup_steps=CFG["warmup_steps"])
    loss_fn = LabelSmoothingLoss(len(train_ds.tgt_vocab), pad_idx, CFG["label_smoothing"])

    best_val = float("inf")

    for epoch in range(CFG["num_epochs"]):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, True, device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, False, device)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch+1}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, "best_checkpoint.pt")

    ckpt = torch.load("best_checkpoint.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    class SimpleVocab:
        def __init__(self, stoi, itos_d):
            self.stoi = stoi
            self.itos = itos_d

    tgt_vocab_obj = SimpleVocab(train_ds.tgt_vocab, train_ds.tgt_itos)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab_obj, device)
    wandb.log({"test_bleu": bleu})
    print(f"Test BLEU: {bleu:.2f}")
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
