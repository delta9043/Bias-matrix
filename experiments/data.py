"""Bias in Bios loading and rule-based gender swaps (CDA); hazard consultations of the Korean 1372 consumer centre."""
import re

import numpy as np
import pandas as pd

from .common import DATA_DIR

REPO = "LabHC/bias_in_bios"
FILES = {
    "train": "data/train-00000-of-00001-0ab65b32c47407e8.parquet",
    "dev": "data/dev-00000-of-00001-e6551072fff26949.parquet",
    "test": "data/test-00000-of-00001-5598c840ce8de1ee.parquet",
}

# Class Index Order of the Occupations (LabHC/bias_in_bios dataset card); gender: 0 male, 1 female
PROFESSIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian", "composer",
    "dentist", "dietitian", "dj", "filmmaker", "interior_designer", "journalist",
    "model", "nurse", "painter", "paralegal", "pastor", "personal_trainer",
    "photographer", "physician", "poet", "professor", "psychologist", "rapper",
    "software_engineer", "surgeon", "teacher", "yoga_teacher",
]


def load_bios():
    """All splits in one frame; columns: uid, split, hard_text, profession, gender."""
    from huggingface_hub import hf_hub_download

    frames = []
    for split, fname in FILES.items():
        path = hf_hub_download(REPO, fname, repo_type="dataset", local_dir=DATA_DIR / "bias_in_bios")
        df = pd.read_parquet(path)
        df.insert(0, "split", split)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "uid", range(len(df)))  # Global Row ID (train, dev, test order)
    return df


# Gendered Word Pairs (male, female); swapped in both directions
_PAIRS = [
    ("he", "she"), ("himself", "herself"), ("mr", "ms"),
    ("man", "woman"), ("men", "women"), ("male", "female"), ("males", "females"),
    ("boy", "girl"), ("boys", "girls"), ("father", "mother"), ("fathers", "mothers"),
    ("son", "daughter"), ("sons", "daughters"), ("husband", "wife"), ("husbands", "wives"),
    ("brother", "sister"), ("brothers", "sisters"), ("dad", "mom"),
    ("grandfather", "grandmother"), ("grandson", "granddaughter"), ("uncle", "aunt"),
    ("nephew", "niece"), ("boyfriend", "girlfriend"), ("actor", "actress"),
    ("king", "queen"), ("prince", "princess"), ("gentleman", "lady"), ("gentlemen", "ladies"),
    ("businessman", "businesswoman"), ("chairman", "chairwoman"), ("spokesman", "spokeswoman"),
]
_SWAP = {m: f for m, f in _PAIRS} | {f: m for m, f in _PAIRS}
_SWAP |= {"him": "her", "his": "her", "hers": "his", "mrs": "mr"}

# Words after Objective "her" (otherwise possessive -> "his")
_OBJ_NEXT = set(
    "a an the to in on at for with from by of about as and or but that if when while because "
    "since until after before so than then into onto over under through during is was are were "
    "has had have will would can could may might should do does did not no this these those "
    "up out off back down away again here there today now too also".split()
)
_WORD = re.compile(r"[A-Za-z]+")
_NEXT_WORD = re.compile(r"\s+([A-Za-z]+)")


def _match_case(src, tgt):
    if src.isupper() and len(src) > 1:
        return tgt.upper()
    return tgt.capitalize() if src[0].isupper() else tgt


def swap_gender(text):
    """Counterfactual text: every gendered word replaced by its counterpart."""

    def repl(m):
        word = m.group(0)
        low = word.lower()
        if low == "her":  # her -> his (possessive) / him (objective)
            nxt = _NEXT_WORD.match(text, m.end())
            tgt = "his" if nxt and nxt.group(1).lower() not in _OBJ_NEXT else "him"
        else:
            tgt = _SWAP.get(low)
        return _match_case(word, tgt) if tgt else word

    return _WORD.sub(repl, text)


# ------------------------------------------------------ hazard consultations
# data.go.kr 15098320 (Fair Trade Commission, 1372 centre): hazard consultations with case bodies, CP949
HAZARD_CSV = DATA_DIR / "consumer" / "hazard_15098320.csv"
_REMOTE = ["24", "27"]  # sales-method codes read as e-commerce and TV home shopping from body keywords (no code table found)


def load_hazard(path=HAZARD_CSV, min_class=1000, min_cell=100, test_share=0.2):
    """Cases sold in general stores (code 10) or remotely (24, 27); columns: uid, split, text, code, cls, channel, method.

    text: title and body.  cls: index of the first two item-code digits (classes with at least min_class rows and
    min_cell rows in each channel).  channel: 1 remote sale.  method: raw sales-method code.  split: fixed random train / test.
    """
    raw = pd.read_csv(path, encoding="cp949", dtype=str, on_bad_lines="skip")  # a few malformed rows carry an extra field
    raw.columns = [c.split("(")[0].strip() for c in raw.columns]
    raw = raw[raw["판매방식코드"].isin(["10"] + _REMOTE)]
    raw = raw.assign(code=raw["품목코드"].str[:2], channel=raw["판매방식코드"].isin(_REMOTE).astype(int))
    cells = raw.groupby("code")["channel"].agg(["size", "sum"])
    keep = cells.index[(cells["size"] >= min_class) & (cells["sum"] >= min_cell) & (cells["size"] - cells["sum"] >= min_cell)]
    raw = raw[raw["code"].isin(keep)]
    df = pd.DataFrame({
        "text": (raw["사건제목"].fillna("") + " " + raw["사건내용"].fillna("")).to_numpy(), "code": raw["code"].to_numpy(),
        "cls": raw["code"].astype("category").cat.codes.to_numpy().astype(int), "channel": raw["channel"].to_numpy(),
        "method": raw["판매방식코드"].to_numpy(),
    })
    rank = np.random.default_rng(0).permutation(len(df))
    df.insert(0, "split", np.where(rank < test_share * len(df), "test", "train"))
    df.insert(0, "uid", range(len(df)))
    return df
