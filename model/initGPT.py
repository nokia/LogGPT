#© 2023 Nokia
#Licensed under the Creative Commons Attribution Non Commercial 4.0 International license
#SPDX-License-Identifier: CC-BY-NC-4.0
#

from utils import vocab, logdataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel
import math
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score


class InitGPT(nn.Module):
    def __init__(self, options, train_df, test_df, topk=None):
        super().__init__()
        self.options = options
        self.vocab_path = './datasets/' + self.options['dataset_name'] + '_vocab.pkl'
        self.dataset_name = options['dataset_name']
        self.sliding_window = options['sliding_window']
        self.train_df = train_df
        self.test_df = test_df
        self.device = options['device']
        self.max_lens = options['max_lens']
        self.init_logGPT = options['init_logGPT']
        self.tqdm = options['tqdm']
        self._build_vocab()
        if topk:
            self.top_k = int((len(self.vocab)-5)*topk)
        else:
            self.top_k = min(int((len(self.vocab)-5)*0.95), options['top_k'])
        self._init_training()
        self._predict_topk(self.test_df['EventSequence'].tolist()[::1], self.test_df['Label'].tolist()[::1])

    def _build_vocab(self):
        # Build vocab
        if self.options['building_vocab']:
            print('Building vocab...')
            text = self.train_df['EventSequence'].tolist()
            self.vocab = vocab.WordVocab(text, sliding_window=self.sliding_window)
            self.vocab.save_vocab(self.vocab_path)
        else:
            print('Loading vocab...')
            self.vocab = vocab.WordVocab.load_vocab(self.vocab_path)
        print(f'Vocab size: {len(self.vocab)}')
        print(self.vocab.itos)

    def _save(self, name=None):
        if name is None:
            torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    }, f'./saved_models/log_gpt2_{self.dataset_name}.pth')
        else:
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optim.state_dict(),
                        }, f'./saved_models/log_gpt2_{self.dataset_name}_{name}.pth')
    def _load(self, name=None):
        if name is None:
            checkpoint = torch.load(f'./saved_models/log_gpt2_{self.dataset_name}.pth', map_location=self.device)
        else:
            checkpoint = torch.load(f'./saved_models/log_gpt2_{self.dataset_name}_{name}.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

    def _init_training(self):
        config = GPT2Config(vocab_size=len(self.vocab),
                            bos_token_id=self.vocab.stoi['<bos>'],
                            eos_token_id=self.vocab.stoi['<eos>'],
                            pad_token_id=self.vocab.stoi['<pad>'],
                            unk_token_id=self.vocab.stoi['<unk>'],
                            mask_token_id=self.vocab.stoi['<mask>'],
                            n_positions=self.max_lens,
                            output_hidden_states=True,
                            n_layer=self.options['n_layers'],
                            n_head=self.options['n_heads'],
                            n_embd=self.options['n_embd']
                            )
        self.model = GPT2LMHeadModel(config)
        self.model.resize_token_embeddings(len(self.vocab))
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.options['init_lr'])
        if self.init_logGPT:
            print('Training Initial LogGPT...')
            valid_index = int(0.9*len(self.train_df))
            train_encodings = [self.vocab.forward(i) for i in self.train_df['EventSequence'].tolist()[:valid_index]]
            train_dataset = logdataset.LogDataset(train_encodings, self.vocab.stoi['<pad>'])
            train_loader = DataLoader(train_dataset, batch_size=self.options['init_batch_size'], shuffle=False,
                                      collate_fn=train_dataset.collate_fn)
            valid_encodings = [self.vocab.forward(i) for i in self.train_df['EventSequence'].tolist()[valid_index:]]
            valid_dataset = logdataset.LogDataset(valid_encodings, self.vocab.stoi['<pad>'])
            valid_loader = DataLoader(valid_dataset, batch_size=self.options['init_batch_size'], shuffle=False,
                                        collate_fn=valid_dataset.collate_fn)
            self.model.to(self.device)
            best_loss = math.inf
            for epoch in tqdm(range(self.options['init_num_epochs']), desc='Epoch:'):
                self.model.train()
                epoch_loss = 0
                for batch in tqdm(train_loader, desc='Training:', disable=not self.tqdm):
                    self.optim.zero_grad()
                    input = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    outputs = self.model(input, attention_mask=attention_mask, labels=input)
                    loss = outputs.loss
                    loss.backward()
                    self.optim.step()

                self.model.eval()
                for batch in tqdm(valid_loader, desc='Validating:', disable=not self.tqdm):
                    with torch.no_grad():
                        input = batch[0].to(self.device)
                        attention_mask = batch[1].to(self.device)
                        outputs = self.model(input, attention_mask=attention_mask, labels=input)
                        loss = outputs.loss
                        epoch_loss += loss.item()
                if epoch_loss/len(valid_loader) < best_loss:
                    best_loss = epoch_loss/len(valid_loader)
                    self._save()
                print(f'Epoch {epoch} loss: {epoch_loss/len(valid_loader)}')

        else:
            print('Loading LogGPT...')
            checkpoint = torch.load(f'./saved_models/log_gpt2_{self.dataset_name}.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            print('LogGPT loaded.')

    def _predict_topk(self, seqs, label, ratio=None):
        self.model.eval()
        y_true = []
        y_pred = []
        print('Predicting with topk...')
        for ind, seq in enumerate(tqdm(seqs, desc='Predicting with topk:', disable=not self.tqdm)):
            y_true.append(label[ind])
            seq_ids = self.vocab.forward(seq)
            lst_ys = seq_ids[1:] if self.sliding_window else seq_ids[2:]
            input_ids = torch.tensor([seq_ids]).to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids).logits[:, :-1, :]
            if not self.sliding_window:
                logits = logits[:, 1:, :]
            probabilities = torch.softmax(logits, dim=-1)
            topk = torch.topk(probabilities, k=self.top_k)
            lst_topk = topk.indices.tolist()[0]
            lst_hit = []
            for ind, val in enumerate(lst_ys):
                if val in lst_topk[ind]:
                    lst_hit.append(1)
                else:
                    lst_hit.append(0)
            if sum(lst_hit) < len(lst_ys):
                y_pred.append(1)
            else:
                y_pred.append(0)
        print('Topk results:')
        print(classification_report(y_true=y_true, y_pred=y_pred, digits=5))
        print(confusion_matrix(y_true=y_true, y_pred=y_pred))
        print(f'AUC ROC: {roc_auc_score(y_true=y_true, y_score=y_pred)}')
        print(F'AUC PR: {average_precision_score(y_true=y_true, y_score=y_pred)}')

    def predict(self, seqs, label, cut=None, result=0):
        self.model.eval()
        if cut is None:
            cut = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        y_pred = []
        y_score = []
        print('Predicting with generated...')
        for seq in tqdm(seqs, desc='Predicting with generated:', disable=not self.tqdm):
            outputs = torch.empty(0)
            seq_ids = self.vocab.forward(seq)
            lens = len(seq_ids)
            for i in cut:
                if math.floor(i*lens) < 2:
                    input_ids = torch.tensor([seq_ids[:2]]).to(self.device)
                elif math.floor(i*lens) >= lens-2:
                    input_ids = torch.tensor([seq_ids[:-2]]).to(self.device)
                else:
                    input_ids = torch.tensor([seq_ids[:math.floor(i*lens)]]).to(self.device)
                outputs = torch.cat((outputs, self.model.generate(input_ids,
                                        max_length=lens,
                                        min_length=lens-1,
                                        # num_beams=self.options['num_return_sequences'],
                                        do_sample=True,
                                        top_k=self.options['top_k'],
                                        pad_token_id=self.vocab.stoi['<pad>'],
                                        num_return_sequences=self.options['num_return_sequences'],
                                        early_stopping=False).cpu()), dim=0)
            check_tensor = torch.tensor(seq_ids).to(torch.int)
            count = 0
            for row in outputs.to(torch.int):
                if torch.all(check_tensor == row):
                    count += 1
            y_score.append(count/len(outputs))
            if count >= 1:
                y_pred.append(0)
            else:
                y_pred.append(1)
        print('Generated results:')
        if result == 1:
            print(classification_report(y_true=label, y_pred=y_pred, digits=5))
            print(confusion_matrix(y_true=label, y_pred=y_pred))
            print(f'AUC ROC: {roc_auc_score(y_true=label, y_score=y_pred)}')
            print(F'AUC PR: {average_precision_score(y_true=label, y_score=y_pred)}')
        return y_pred, y_score