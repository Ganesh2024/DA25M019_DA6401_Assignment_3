import spacy
from datasets import load_dataset
from collections import Counter


class Multi30kDataset:
    UNK, PAD, SOS, EOS = "<unk>", "<pad>", "<sos>", "<eos>"
    SPECIALS = [UNK, PAD, SOS, EOS]

    def __init__(self, split="train"):
        self.split = split
        self.data = load_dataset("bentrevett/multi30k", split=split)

        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            from spacy.cli import download as spacy_dl
            spacy_dl("de_core_news_sm")
            self.spacy_de = spacy.load("de_core_news_sm")

        try:
            self.spacy_en = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download as spacy_dl
            spacy_dl("en_core_web_sm")
            self.spacy_en = spacy.load("en_core_web_sm")

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self, train_data=None, min_freq=2):
        if train_data is None:
            train_data = self.data

        de_counter, en_counter = Counter(), Counter()
        for example in train_data:
            for tok in self.tokenize_de(example["de"]):
                de_counter[tok] += 1
            for tok in self.tokenize_en(example["en"]):
                en_counter[tok] += 1

        def make_vocab(counter, min_freq):
            vocab = {tok: idx for idx, tok in enumerate(self.SPECIALS)}
            for tok, freq in counter.items():
                if freq >= min_freq and tok not in vocab:
                    vocab[tok] = len(vocab)
            itos = {v: k for k, v in vocab.items()}
            return vocab, itos

        self.src_vocab, self.src_itos = make_vocab(de_counter, min_freq)
        self.tgt_vocab, self.tgt_itos = make_vocab(en_counter, min_freq)
        return self.src_vocab, self.tgt_vocab

    def encode(self, tokens, vocab):
        unk = vocab[self.UNK]
        sos = vocab[self.SOS]
        eos = vocab[self.EOS]
        return [sos] + [vocab.get(t, unk) for t in tokens] + [eos]

    def process_data(self):
        processed = []
        for example in self.data:
            src_tok = self.tokenize_de(example["de"])
            tgt_tok = self.tokenize_en(example["en"])
            src_ids = self.encode(src_tok, self.src_vocab)
            tgt_ids = self.encode(tgt_tok, self.tgt_vocab)
            processed.append((src_ids, tgt_ids))
        return processed
