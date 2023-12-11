#© 2023 Nokia
#Licensed under the Creative Commons Attribution Non Commercial 4.0 International license
#SPDX-License-Identifier: CC-BY-NC-4.0
#

from utils import vocab
import numpy as np
import torch
from tqdm import tqdm
import copy
import math
from torch.distributions import Categorical


class LogGPT(object):
    def __init__(self, options, train_df, test_df, initGPT_model, topk=None, cut=None):
        self.options = options
        self.vocab_path = './datasets/' + self.options['dataset_name'] + '_vocab.pkl'
        self.train_index = int(0.9 * len(train_df))
        self.train_df = train_df[:self.train_index]
        self.val_df = train_df[self.train_index:]
        self.test_df = test_df
        self.save_memory = options['save_memory']
        self.tqdm = options['tqdm']
        self.device = options['device']
        self.logGPT_lr = options['logGPT_lr']
        self.FT_GPT = copy.deepcopy(initGPT_model)
        self.FT_GPT.optim = torch.optim.Adam(self.FT_GPT.model.parameters(), lr=self.logGPT_lr)
        self.initGPT = initGPT_model
        if cut:
            self.cut = cut
        else:
            self.cut = [0.2]
        self.epsilon = 1e-7
        self.logGPT_episode = options['logGPT_episode']
        self._load_vocab()
        if topk:
            self.top_k = int((len(self.vocab)-5)*topk)
        else:
            self.top_k = min(int((len(self.vocab)-5)*0.95), options['top_k'])
        self.logGPT_training = options['logGPT_training']
        if self.logGPT_training:
            self.train()
            self.test()
        else:
            self.test()

    def _load_vocab(self):
        print('Loading vocab...')
        self.vocab = vocab.WordVocab.load_vocab(self.vocab_path)
        print(f'Vocab size: {len(self.vocab)}')
        print(self.vocab.itos)

    def _compute_log_prob(self, sequence):
        # Run the model on the sequence
        outputs = self.FT_GPT.model(sequence)
        # Compute the log probabilities of the generated tokens
        log_probs = []
        for i in range(1, sequence.size(1)):
            logits = outputs.logits[0, i - 1, :]
            distribution = Categorical(logits=logits)
            log_prob = distribution.log_prob(sequence[0, i])
            log_probs.append(log_prob)
        # Sum the log probabilities to get the log probability of the sequence
        return sum(log_probs)

    def step(self, seq):
        # 1. generate samples by FT_GPT
        generated_samples = torch.empty(0)
        seq_ids = self.vocab.forward(seq)
        lens = len(seq_ids)
        self.FT_GPT.model.train()
        self.FT_GPT.optim.zero_grad()
        for i in self.cut:
            if self.options['sliding_window']:
                if math.floor(i*lens) < 1:
                    input_ids = torch.tensor([seq_ids[:1]]).to(self.device)
                elif math.floor(i*lens) >= lens-1:
                    input_ids = torch.tensor([seq_ids[:-1]]).to(self.device)
                else:
                    input_ids = torch.tensor([seq_ids[:math.floor(i*lens)]]).to(self.device)
            else:
                if math.floor(i*lens) < 2:
                    input_ids = torch.tensor([seq_ids[:2]]).to(self.device)
                elif math.floor(i*lens) >= lens-2:
                    input_ids = torch.tensor([seq_ids[:-2]]).to(self.device)
                else:
                    input_ids = torch.tensor([seq_ids[:math.floor(i*lens)]]).to(self.device)
            generated_samples = torch.cat((generated_samples, self.FT_GPT.model.generate(input_ids, max_length=lens, min_length=lens,
                                    num_beams=self.options['num_return_sequences'], do_sample=True, pad_token_id=self.vocab.stoi['<pad>'],
                                    num_return_sequences=1,
                                    early_stopping=False).cpu()), dim=0)

        generated_samples = generated_samples.to(self.device).to(torch.long)
        # 2. compute rewards
        losses = self.compute_loss(seq_ids, generated_samples)

        # 3. update FT_GPT
        losses.backward()
        self.FT_GPT.optim.step()

    def train(self):
        self.initGPT.eval()
        self.initGPT.to(self.device)
        self.FT_GPT.to(self.device)
        # initialize loss
        init_loss = 0
        for seq in (self.val_df['EventSequence'].tolist()):
            init_loss += self.valid_step(seq)
        best_loss = init_loss/len(self.val_df)
        print(f'Initial loss: {best_loss.item()}')
        count = 0
        last_episode_loss = init_loss/len(self.val_df)
        for episode in tqdm(range(self.logGPT_episode), desc='LogGPT training:'):
            episode_loss = 0
            random_seed = np.random.randint(0, 100)
            np.random.seed(random_seed)
            shuffled_index = np.random.permutation(len(self.train_df))[:int(len(self.train_df)*1.0)]
            for seq in tqdm(self.train_df.iloc[shuffled_index]['EventSequence'].tolist(), desc='LogGPT training steps:',
                            disable=not self.tqdm):
                self.step(seq)
            for seq in tqdm(self.val_df['EventSequence'].tolist(), desc='LogGPT validation steps:',
                            disable=not self.tqdm):
                episode_loss += self.valid_step(seq)
            current_episode_loss = episode_loss/len(self.val_df)
            if current_episode_loss <= best_loss or episode == 0:
                best_loss = current_episode_loss
                self.FT_GPT._save('FT')
            if (last_episode_loss.item()-current_episode_loss.item())/(current_episode_loss.item()+self.epsilon) < 0.01:
                count += 1
            last_episode_loss = current_episode_loss
            if count > 3:
                print('Early stop!')
                break
            print(f'Episode {episode} loss: {current_episode_loss.item()}')

            self.FT_GPT._predict_topk(self.test_df['EventSequence'].tolist()[::1], self.test_df['Label'].tolist()[::1])


    def test(self):
        self.FT_GPT._load('FT')
        self.FT_GPT._predict_topk(self.test_df['EventSequence'].tolist()[::1],
                                    self.test_df['Label'].tolist()[::1])

    def valid_step(self, seq):
        # 1. generate samples by FT_GPT
        generated_samples = torch.empty(0)
        seq_ids = self.vocab.forward(seq)
        lens = len(seq_ids)
        self.FT_GPT.model.eval()
        with torch.no_grad():
            for i in self.cut:
                if self.options['sliding_window']:
                    if math.floor(i * lens) < 1:
                        input_ids = torch.tensor([seq_ids[:1]]).to(self.device)
                    elif math.floor(i * lens) >= lens - 1:
                        input_ids = torch.tensor([seq_ids[:-1]]).to(self.device)
                    else:
                        input_ids = torch.tensor([seq_ids[:math.floor(i * lens)]]).to(self.device)
                else:
                    if math.floor(i * lens) < 2:
                        input_ids = torch.tensor([seq_ids[:2]]).to(self.device)
                    elif math.floor(i * lens) >= lens - 2:
                        input_ids = torch.tensor([seq_ids[:-2]]).to(self.device)
                    else:
                        input_ids = torch.tensor([seq_ids[:math.floor(i * lens)]]).to(self.device)
                generated_samples = torch.cat(
                    (generated_samples, self.FT_GPT.model.generate(input_ids, max_length=lens, min_length=lens,
                                                                   num_beams=self.options['num_return_sequences'],
                                                                   do_sample=True, pad_token_id=self.vocab.stoi['<pad>'],
                                                                   num_return_sequences=1,
                                                                   early_stopping=False).cpu()), dim=0)

            # change the data type of generated_samples from torch.float to torch.long
            generated_samples = generated_samples.to(self.device).to(torch.long)
            # 2. compute rewards
            losses = self.compute_loss(seq_ids, generated_samples)
        return losses


    def compute_loss(self, seq_ids, generated_samples):
        # compute topk reward
        topk_reward = 0
        log_prob = 0
        outputs = self.FT_GPT.model(generated_samples)
        logits = outputs.logits.cpu()[:, :-1]
        logits = logits[0] if self.options['sliding_window'] else logits[0, 1:]
        lst_ys = torch.tensor(seq_ids).unsqueeze(0)
        lst_ys = lst_ys[0, 1:] if self.options['sliding_window'] else lst_ys[0, 2:]
        for i in range(len(lst_ys)):
            distribution = Categorical(logits=logits[i])
            log_prob += distribution.log_prob(lst_ys[i])
            topk_preds = logits[i].topk(self.top_k, dim=-1).indices
            if lst_ys[i] in topk_preds:
                topk_reward += 1
            else:
                topk_reward -= 1

        loss_topk = -log_prob * topk_reward
        loss_topk = loss_topk.to(self.device)

        return loss_topk

