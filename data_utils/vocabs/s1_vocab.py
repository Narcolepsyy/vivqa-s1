import os
from collections import Counter
from typing import List

import torch

from builders.vocab_builder import META_VOCAB
from data_utils.utils import preprocess_sentence
from data_utils.vocabs.vocab import Vocab
from utils.viso_s1 import read_jsonl

CAND_MAX_TOKENS = 6


@META_VOCAB.register()
class S1Vocab(Vocab):
    """Vocabulary for ViVQA-S1 typed decisions.

    Words come from the TRAIN + CALIB questions and the candidate texts of the choice items in TRAIN + CALIB
    (dev/test words that are not in it become <unk>; their LENGTHS are used for max_question_length only).
    The flat answer vocabulary (baseline A0 only): <yes>, <no>, #0..#10 and every distinct choice gold_label of
    the TRAIN split; `total_answers` is its size. Vocab.__init__ is not called (it reads ViVQA-style json and
    would use dev/test words), the same attributes are set here instead."""

    def __init__(self, config):
        self.tokenizer = config.TOKENIZER
        self.padding_token, self.bos_token = config.PAD_TOKEN, config.BOS_TOKEN
        self.eos_token, self.unk_token = config.EOS_TOKEN, config.UNK_TOKEN
        d = config.S1_DIR
        files = {n: os.path.join(d, f"{n}.jsonl") for n in ("train", "calib", "dev", "test")}

        self.freqs, self.max_question_length, self.max_answer_length = Counter(), 0, 0
        flat = set()
        for name in ("train", "calib", "dev", "test"):
            for it in read_jsonl(files[name]):
                q = it["questions"]["q"]
                qtoks = preprocess_sentence(q["instructions"], self.tokenizer)
                self.max_question_length = max(self.max_question_length, len(qtoks) + 2)
                if name in ("train", "calib"):
                    self.freqs.update(qtoks)
                    if q["type"] == "choice":
                        for c in q["criteria"]:
                            self.freqs.update(preprocess_sentence(c, self.tokenizer))
                if name == "train" and q["type"] == "choice":
                    flat.add(it["gold_label"])

        specials = [self.padding_token, self.bos_token, self.eos_token, self.unk_token]
        counter = Counter({w: n for w, n in self.freqs.items() if w not in specials})
        words = sorted(counter.items(), key=lambda t: t[0])
        words.sort(key=lambda t: t[1], reverse=True)
        itos = specials + [w for w, n in words if n >= max(config.MIN_FREQ, 1)]
        self.itos = {i: t for i, t in enumerate(itos)}
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.specials = specials
        self.padding_idx, self.bos_idx = self.stoi[self.padding_token], self.stoi[self.bos_token]
        self.eos_idx, self.unk_idx = self.stoi[self.eos_token], self.stoi[self.unk_token]
        self.word_embeddings = None

        self.flat_itos = ["<yes>", "<no>"] + [f"#{i}" for i in range(11)] + sorted(flat)
        self.flat_stoi = {t: i for i, t in enumerate(self.flat_itos)}
        self.total_answers = len(self.flat_itos)

    def encode_candidate(self, text: str, max_tokens: int = CAND_MAX_TOKENS) -> List[int]:
        """Word ids of a candidate text (no bos/eos), truncated to max_tokens."""
        toks = preprocess_sentence(text, self.tokenizer)[:max_tokens]
        return [self.stoi.get(t, self.unk_idx) for t in toks]

    def flat_index(self, qtype: str, key: str) -> int:
        """Index in the flat answer vocab of a label key (-1 if absent): noul false->'<no>', true->'<yes>';
        score level k -> '#k'; choice -> the candidate text itself."""
        if qtype == "noul":
            name = "<yes>" if key == "true" else "<no>"
        elif qtype == "score":
            name = f"#{key}"
        else:
            name = key
        return self.flat_stoi.get(name, -1)
