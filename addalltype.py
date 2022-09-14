cfg = {
    "num_proc": 1,
    # data
    "k_folds": 5,
    "max_length": 2048,
    "padding": False,
    "stride": 0,
    "data_dir": "feedback-prize-effectiveness",
    "load_from_disk": None, # if you already tokenized, you can load it through this
    "pad_multiple": 512,
    # model
    "model_name_or_path": "allenai/longformer-base-4096",
    "dropout": 0.1,
    # to put in TrainingArguments
    "trainingargs": {
        "output_dir": "output",
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 4,
        "learning_rate": 9e-6,
        "weight_decay": 0.01,
        "num_train_epochs": 3,
        "warmup_ratio": 0.1,
        "optim": 'adamw_torch',
        "logging_steps": 50,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "report_to": "none",
        "group_by_length": True,
        "save_total_limit": 1,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "seed": 18,
        # you should probably set "fp16" to True, but it doesn't really matter on Kaggle
    }
}



import re
import pickle
import codecs
import warnings
import logging
from functools import partial
from pathlib import Path
from itertools import chain
from text_unidecode import unidecode
from typing import Any, Optional, Tuple

import pandas as pd
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, set_seed

from datasets import Dataset, load_from_disk

# https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313330
def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def read_text_files(example, data_dir):

    id_ = example["essay_id"]

    with open(data_dir / "train" / f"{id_}.txt", "r") as fp:
        example["text"] = resolve_encodings_and_normalize(fp.read())

    return example

set_seed(cfg["trainingargs"]["seed"])

# change logging to not be bombarded by messages
# if you are debugging, the messages will likely be helpful
warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

data_dir = Path(cfg["data_dir"])

if cfg["load_from_disk"]:
    if not cfg["load_from_disk"].endswith(".dataset"):
        cfg["load_from_disk"] += ".dataset"
    ds = load_from_disk(cfg['load_from_disk'])

    pkl_file = f"{cfg['load_from_disk'][:-len('.dataset')]}_pkl"
    with open(pkl_file, "rb") as fp:
        grouped = pickle.load(fp)

    print("Loading from saved files")
else:
    train_df = pd.read_csv(data_dir / "train.csv")

    text_ds = Dataset.from_dict({"essay_id": train_df.essay_id.unique()})

    text_ds = text_ds.map(
        partial(read_text_files, data_dir=data_dir),
        num_proc=cfg["num_proc"],
        batched=False,
        desc="Loading text files",
    )

    text_df = text_ds.to_pandas()

    train_df["discourse_text"] = [
        resolve_encodings_and_normalize(x) for x in train_df["discourse_text"]
    ]

    train_df = train_df.merge(text_df, on="essay_id", how="left")

disc_types = [
    "Claim",
    "Concluding Statement",
    "Counterclaim",
    "Evidence",
    "Lead",
    "Position",
    "Rebuttal",
]
cls_tokens_map = {label: f"[CLS_{label.upper()}]" for label in disc_types}
end_tokens_map = {label: f"[END_{label.upper()}]" for label in disc_types}

label2id = {
    "Adequate": 0,
    "Effective": 1,
    "Ineffective": 2,
}

tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
tokenizer.add_special_tokens(
    {"additional_special_tokens": list(cls_tokens_map.values()) + list(end_tokens_map.values())}
)
cls_id_map = {
    label: tokenizer.encode(tkn)[1]
    for label, tkn in cls_tokens_map.items()
}


def find_positions(example):
    text = example["text"][0]

    # keeps track of what has already
    # been located
    min_idx = 0

    # stores start and end indexes of discourse_texts
    idxs = []

    for dt in example["discourse_text"]:
        # calling strip is essential
        matches = list(re.finditer(re.escape(dt.strip()), text))

        # If there are multiple matches, take the first one
        # that is past the previous discourse texts.
        if len(matches) > 1:
            for m in matches:
                if m.start() >= min_idx:
                    break
        # If no matches are found
        elif len(matches) == 0:
            idxs.append([-1])  # will filter out later
            continue
            # If one match is found
        else:
            m = matches[0]

        idxs.append([m.start(), m.end()])

        min_idx = m.start()

    return idxs


def tokenize(example):
    example["idxs"] = find_positions(example)

    text = example["text"][0]
    chunks = []
    labels = []
    prev = 0

    zipped = zip(
        example["idxs"],
        example["discourse_type"],
        example["discourse_effectiveness"],
    )
    for idxs, disc_type, disc_effect in zipped:
        # when the discourse_text wasn't found
        if idxs == [-1]:
            continue

        s, e = idxs

        # if the start of the current discourse_text is not
        # at the end of the previous one.
        # (text in between discourse_texts)
        if s != prev:
            chunks.append(text[prev:s])
            prev = s

        # if the start of the current discourse_text is
        # the same as the end of the previous discourse_text
        if s == prev:
            chunks.append(cls_tokens_map[disc_type])
            chunks.append(text[s:e])
            chunks.append(end_tokens_map[disc_type])

        prev = e

        labels.append(label2id[disc_effect])
    a=" ".join(chunks)
    tokenized = tokenizer(
        " ".join(chunks),
        padding=False,
        truncation=True,
        max_length=cfg["max_length"],
        add_special_tokens=True,
    )

    # at this point, labels is not the same shape as input_ids.
    # The following loop will add -100 so that the loss function
    # ignores all tokens except CLS tokens

    # idx for labels list
    idx = 0
    final_labels = []
    for id_ in tokenized["input_ids"]:
        # if this id belongs to a CLS token
        if id_ in cls_id_map.values():
            final_labels.append(labels[idx])
            idx += 1
        else:
            # -100 will be ignored by loss function
            final_labels.append(-100)

    tokenized["labels"] = final_labels

    return tokenized


# I frequently restart my notebook, so to reduce time
# you can set this to just load the tokenized dataset from disk.
# It gets loaded in the 3rd code cell, but a check is done here
# to skip tokenizing
if cfg["load_from_disk"] is None:
    # make lists of discourse_text, discourse_effectiveness
    # for each essay
    grouped = train_df.groupby(["essay_id"]).agg(list)
    a=grouped["text"][0]
    ds = Dataset.from_pandas(grouped)

    ds = ds.map(
        tokenize,
        batched=False,
        num_proc=cfg["num_proc"],
        desc="Tokenizing",
    )

    save_dir = f"{cfg['trainingargs']['output_dir']}"
    ds.save_to_disk(f"{save_dir}.dataset")
    with open(f"{save_dir}_pkl", "wb") as fp:
        pickle.dump(grouped, fp)
    print("Saving dataset to disk:", cfg['trainingargs']['output_dir'])


# basic kfold
def get_folds(df, k_folds=5):
    kf = KFold(n_splits=k_folds)
    return [
        val_idx
        for _, val_idx in kf.split(df)
    ]


fold_idxs = get_folds(ds["labels"], cfg["k_folds"])