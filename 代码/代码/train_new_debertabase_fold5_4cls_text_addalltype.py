# %%
import os
import gc
import copy
import time
import random
import string
import joblib

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
# Utils
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import f1_score, log_loss
# For Transformer Models
from transformers import AutoTokenizer, AutoModel, AutoConfig, AdamW
from transformers import DataCollatorWithPadding

# For colored terminal text
from colorama import Fore, Back, Style
import re
from datasets import Dataset
b_ = Fore.BLUE
y_ = Fore.YELLOW
sr_ = Style.RESET_ALL

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# %%
import wandb

try:
    from kaggle_secrets import UserSecretsClient

    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("wandb_api")
    wandb.login(key=api_key)
    anony = None
except:
    anony = "must"
    print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your '
          'W&B access token. Use the Label name as wandb_api. \nGet your W&B access token '
          'from here: https://wandb.ai/authorize')


# %%
def id_generator(size=12, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))


HASH_NAME = id_generator(size=12)
print(HASH_NAME)
# %%
TRAIN_DIR = "feedback-prize-effectiveness/train"
TEST_DIR = "feedback-prize-effectiveness/test"
# %%
CONFIG = {"seed": 42,
          "epochs": 3,
          "model_name": "microsoft/deberta-v3-base",
          "train_batch_size": 8,
          "valid_batch_size": 16,
          "max_length": 512,
          "learning_rate": 1e-5,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-6,
          "T_max": 500,
          "weight_decay": 1e-6,
          "n_fold": 5,
          "n_accumulate": 1,
          "num_classes": 3,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          "hash_name": HASH_NAME,
          "competition": "FeedBack",
          "_wandb_kernel": "deb",
          "fc_dropout":0.2,
          "gradient_checkpoint" : False,
          }
# num_workers = 1
# path = "../input/feedback-deberta-large-051/"
# config_path = path + 'config.pth'
# model = "microsoft/deberta-large"
# batch_size = 16
# fc_dropout = 0.2
# target_size = 3
# max_len = 512
# seed = 42
# n_fold = 4
# trn_fold = [i for i in range(n_fold)]
# gradient_checkpoint = False
CONFIG["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG['model_name'])
CONFIG['group'] = f'{HASH_NAME}-Baseline'


# %%
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(CONFIG['seed'])


# %%


def get_essay(essay_id):
    essay_path = os.path.join(TRAIN_DIR, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text


# %%
# ====================================================
# Utils
# ====================================================



def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def get_score(y_true, y_pred):
    y_pred = softmax(y_pred)
    score = log_loss(y_true, y_pred)
    return round(score, 5)
df = pd.read_csv("feedback-prize-effectiveness/train.csv")
df['essay_text'] = df['essay_id'].apply(get_essay)
df.head()

# %%
# %%

gkf = GroupKFold(n_splits=CONFIG['n_fold'])

for fold, (_, val_) in enumerate(gkf.split(X=df, groups=df.essay_id)):
    df.loc[val_, "kfold"] = int(fold)

df["kfold"] = df["kfold"].astype(int)
df.head()

# %%

df.groupby('kfold')['discourse_effectiveness'].value_counts()

# %%
from text_unidecode import unidecode
from typing import Dict, List, Tuple
import codecs

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
df['discourse_text'] = df['discourse_text'].apply(lambda x : resolve_encodings_and_normalize(x))
df['essay_text'] = df['essay_text'].apply(lambda x : resolve_encodings_and_normalize(x))
df['text'] = df['essay_text']
encoder = LabelEncoder()
df['discourse_effectiveness'] = encoder.fit_transform(df['discourse_effectiveness'])

with open("le.pkl", "wb") as fp:
    joblib.dump(encoder, fp)



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




    # at this point, labels is not the same shape as input_ids.
    # The following loop will add -100 so that the loss function
    # ignores all tokens except CLS tokens









    a=" ".join(chunks)
    example["text"][0]=a








    return example
# %%
grouped = df.groupby(["essay_id"]).agg(list)

num=range(0,grouped.shape[0])
grouped["num"]=num
ds = Dataset.from_pandas(grouped)
ds = ds.map(
    tokenize,
    batched=False,

)
x=ds["text"]

for i in range(df.shape[0]):
    a1=grouped.loc[df["essay_id"][i]]["num"]
    b1=x[a1][0]
    df["text"][i]=b1

del ds,grouped,x

print(df)


class FeedBackDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.discourse = df['discourse_text'].values
        self.essay = df['essay_text'].values
        self.essay_type = df['text'].values
        self.targets = df['discourse_effectiveness'].values
        self.discourse_type = df['discourse_type'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        discourse = self.discourse[index]
        discourse_type = self.discourse_type[index]
        essay = self.essay[index]
        essay_type = self.essay_type[index]
        # text = discourse_type+self.tokenizer.sep_token+discourse +self.tokenizer.sep_token + " " + essay
        text = discourse_type+' '+discourse +self.tokenizer.sep_token+essay_type
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': self.targets[index]
        }


#%%
class Collate:
    def __init__(self, tokenizer, isTrain=True):
        self.tokenizer = tokenizer
        self.isTrain = isTrain
        # self.args = args

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]
        if self.isTrain:
            output["target"] = [sample["target"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [s + (batch_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["input_ids"]]
            output["attention_mask"] = [s + (batch_max - len(s)) * [0] for s in output["attention_mask"]]
        else:
            output["input_ids"] = [(batch_max - len(s)) * [self.tokenizer.pad_token_id] + s for s in output["input_ids"]]
            output["attention_mask"] = [(batch_max - len(s)) * [0] + s for s in output["attention_mask"]]

        # convert to tensors
        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(output["attention_mask"], dtype=torch.long)
        if self.isTrain:
            output["target"] = torch.tensor(output["target"], dtype=torch.long)

        return output

collate_fn = Collate(tokenizer=CONFIG['tokenizer'], isTrain=True)
#collate_fn = DataCollatorWithPadding(tokenizer=CONFIG['tokenizer'])
# %%

class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        mean_pooling_embeddings = torch.mean(last_hidden_state, 1)
        _, max_pooling_embeddings = torch.max(last_hidden_state, 1)
        mean_max_embeddings = torch.cat((mean_pooling_embeddings, max_pooling_embeddings), 1)
        return mean_max_embeddings


class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, all_hidden_states):
        ## forward
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = self.dropout(out[:, -1, :])
        return out


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average
# %%
from torch.cuda.amp import autocast
class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.cfg = CONFIG
        self.config = AutoConfig.from_pretrained(model_name,output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name,config=self.config)
        # gradient checkpointing  梯度检查点
        if CONFIG['gradient_checkpoint']:
            self.model.gradient_checkpointing_enable()
            print(f"Gradient Checkpointing: {self.model.is_gradient_checkpointing}")
        self.bilstm = nn.LSTM(self.config.hidden_size, (self.config.hidden_size) // 2, num_layers=2,
                              dropout=self.config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)

        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        #-----------------
        #self.pooler=WeightedLayerPooling(self.config.num_hidden_layers)
        #self.pooler=MeanPooling()

        self.output = nn.Sequential(
            nn.Linear(self.config.hidden_size*4, CONFIG['num_classes'])
            # nn.Linear(256, self.cfg.target_size)
        )

    def loss(self, outputs, targets):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(outputs, targets)
        return loss

    def monitor_metrics(self, outputs, targets):
        device = targets.get_device()
        # print(outputs)
        # print(targets)
        mll = log_loss(
            targets.cpu().detach().numpy(),
            softmax(outputs.cpu().detach().numpy()),
            labels=[0, 1, 2],
        )
        return mll

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # def forward(self, ids, mask):
    #     out = self.model(input_ids=ids, attention_mask=mask,
    #                      output_hidden_states=False)
    #     out = self.pooler(out.last_hidden_state, mask)
    #     out = self.drop(out)
    #     outputs = self.fc(out)
    #     return outputs
    def forward(self, ids, mask, token_type_ids=None, targets=None):
        if token_type_ids:
            transformer_out = self.model(ids, mask, token_type_ids)
        else:
            transformer_out = self.model(ids, mask)

        # LSTM/GRU header
        # all_hidden_states = torch.stack(transformer_out[1])
        # sequence_output = self.pooler(all_hidden_states)3
        all_hidden_states = torch.stack(transformer_out[1])

        concatenate_pooling = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]), -1
        )
        sequence_output = concatenate_pooling[:, 0]
        # simple CLS
        #transformer_out = self.pooler(transformer_out.last_hidden_state,mask)
        #sequence_output = transformer_out
        #sequence_output = transformer_out[0][:, 0, :]
        #all_hidden_states = torch.stack(transformer_out[1])
        #sequence_output=self.pooler(all_hidden_states)

        # Main task
        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))
        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5

        if targets is not None:
            metric = self.monitor_metrics(logits, targets)
            return logits

        return logits


# %%

def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)


# %%

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)
        loss = loss / CONFIG['n_accumulate']
        loss.backward()

        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()

    return epoch_loss


# %%

@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.long)

        batch_size = ids.size(0)

        outputs = model(ids, mask)

        loss = criterion(outputs, targets)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])

    gc.collect()

    return epoch_loss


# %%

def run_training(model, optimizer, scheduler, device, num_epochs, fold):
    # To automatically log gradients
    wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler,
                                           dataloader=train_loader,
                                           device=CONFIG['device'], epoch=epoch)

        val_epoch_loss = valid_one_epoch(model, valid_loader, device=CONFIG['device'],
                                         epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)

        # Log the metrics
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})

        # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(f"{b_}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            run.summary["Best Loss"] = best_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"Loss-Fold-{fold}.bin"
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


# %%

def prepare_loaders(fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_dataset = FeedBackDataset(df_train, tokenizer=CONFIG['tokenizer'], max_length=CONFIG['max_length'])
    valid_dataset = FeedBackDataset(df_valid, tokenizer=CONFIG['tokenizer'], max_length=CONFIG['max_length'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'], collate_fn=collate_fn,
                              num_workers=0, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], collate_fn=collate_fn,
                              num_workers=0, shuffle=False, pin_memory=True)

    return train_loader, valid_loader


# %%

def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'],
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG['T_0'],
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None

    return scheduler


# %%

if __name__ == '__main__':
    torch.cuda.empty_cache()
    for fold in range(0, CONFIG['n_fold']):
        print(f"{y_}====== Fold: {fold} ======{sr_}")
        run = wandb.init(project='FeedBack',
                         config=CONFIG,
                         job_type='Train',
                         group=CONFIG['group'],
                         tags=[CONFIG['model_name'], f'{HASH_NAME}'],
                         name=f'{HASH_NAME}-fold-{fold}',
                         anonymous='must')

        # Create Dataloaders
        train_loader, valid_loader = prepare_loaders(fold=fold)

        model = FeedBackModel(CONFIG['model_name'])
        model.to(CONFIG['device'])

        # Define Optimizer and Scheduler
        optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
        scheduler = fetch_scheduler(optimizer)

        model, history = run_training(model, optimizer, scheduler,
                                      device=CONFIG['device'],
                                      num_epochs=CONFIG['epochs'],
                                      fold=fold)

        run.finish()

        del model, history, train_loader, valid_loader
        _ = gc.collect()
        print()
