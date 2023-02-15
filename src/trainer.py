import torch
import torch.nn as nn
import numpy as np
import gym
import os
from transformers import AutoConfig,AdamW, get_linear_schedule_with_warmup, AutoModelWithLMHead, BertModel
from src.dataloader import all_labels, pad_token_label_id
from src.conll2002_measure import *
from transformers.models.bart.modeling_bart import shift_tokens_right


class Trainer(object):
    def __init__(self, model, params):
        super().__init__()
        self.params = params
        self.dump_path = params.dump_path
        self.model = model
        self.task = params.task
        self.reward_mode = params.reward_mode
        self.rand_sel_ratio = params.rand_sel_ratio
        self.early_stop = params.early_stop
        if self.task =='conversation':
            self.configure_optimizers()
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def training_step(self, X, y):
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(X)
        
        if self.task == 'ner':
            y = y.view(y.size(0)*y.size(1))
            preds = preds.view(preds.size(0)*preds.size(1), preds.size(2))
        
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(preds, y)
        loss.backward()
        self.optimizer.step()
        
        
        if self.reward_mode == 'pred_entropy':
            sig_preds = torch.nn.functional.softmax(preds, dim=1)
            if self.task == 'ner':
                h = -sum([p * torch.log2(p) for all_probs in sig_preds for p in all_probs])
            else:
                h = sum([-(p0 * torch.log2(p0) + (p1)*torch.log2(p1)) for (p0,p1) in sig_preds])
            return loss, acc, h

        return loss, acc
        

    def validation_step(self, X, y):
        self.model.eval()
        preds = self.model(X)
        if self.task == 'ner':
            y = y.view(y.size(0)*y.size(1))
            preds = preds.view(preds.size(0)*preds.size(1), preds.size(2))
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(preds, y)
        return loss, acc

    def accuracy(self, pred, label):
        batch_size, _ = pred.shape
        if self.task == 'ner':
            predicted_classes = pred.argmax(dim = 1, keepdim = True)
            correct_predictions = predicted_classes.squeeze(1).eq(label).sum()
        elif self.task == 'sentiment':
            predicted_classes = pred.argmax(dim=-1)
            correct_predictions = predicted_classes.eq(label).sum()
        accuracy = correct_predictions / batch_size
        return accuracy
    


    def evaluate_ner(self, dataloader):
        self.model.eval()

        pred_list = []
        y_list = []
        for _, X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            y_list.extend(y.data.cpu().numpy()) # y is a list
            X = X.cuda()
            preds = self.model(X)
            pred_list.extend(preds.data.cpu().numpy())
        
        # concatenation
        pred_list = np.concatenate(pred_list, axis=0)   # (length, num_tag)
        pred_list = np.argmax(pred_list, axis=1)
        y_list = np.concatenate(y_list, axis=0)
        
        # calcuate f1 score
        pred_list = list(pred_list)
        y_list = list(y_list)
        lines = []
        for pred_index, gold_index in zip(pred_list, y_list):
            gold_index = int(gold_index)
            if gold_index != pad_token_label_id:
                pred_token = all_labels[pred_index]
                gold_token = all_labels[gold_index]
                lines.append("w" + " " + pred_token + " " + gold_token)
        results = conll2002_measure(lines)
        f1 = results["fb1"]
        return f1

    def evaluate_sentiment(self, dataloader):
        total_loss = 0
        total_acc = 0
        for _, X, y in dataloader:
            X, y = X.cuda(), y.cuda()
            loss, acc = self.validation_step(X, y)
            total_loss += loss.item()
            total_acc += acc.item()
        return total_loss/len(dataloader), total_acc/len(dataloader)
    
    def configure_optimizers(self):
        # optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.01
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            },
        ]
        betas = tuple(map(float, self.params.adam_betas[1:-1].split(',')))
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=betas,
            eps=1e-8,
            lr=3e-5
        )

        # scheduler
        num_training_steps = (
            (self.params.bag_size//self.params.batch_size)
            // self.params.accumulate_grad_batches
            * 1
        )
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.params.num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return
    
    def conversation_test_step(self, batch):
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()

        # https://huggingface.co/blog/how-to-generate
        beam_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        preds = [
            self.tokenizer.decode(beam_output, skip_special_tokens=True)
            for beam_output in beam_outputs
        ]
        with open(os.path.join(self.dump_path, 'preds.txt'), 'w') as f:
            for output in preds:
                f.write('\n'.join(output) + '\n')
        return preds
    def conversation_training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        pad_token_id = self.model.tokenizer.pad_token_id

        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()

        decoder_input_ids = shift_tokens_right(labels, pad_token_id)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
        logits = outputs[0]
        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=lprobs,
            target=labels,
            epsilon=0.1,
            ignore_index=pad_token_id
        )
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss
    
    def conversation_validation_step(self, batch):
        self.model.eval()
        pad_token_id = self.model.tokenizer.pad_token_id

        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()

        decoder_input_ids = shift_tokens_right(labels, pad_token_id)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
        logits = outputs[0]
        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=lprobs,
            target=labels,
            epsilon=0.1,
            ignore_index=pad_token_id
        )
        return loss
    def evaluate_conversation_model(self, dataloader):
        total_loss = 0
        for batch in dataloader:
            loss = self.conversation_validation_step(batch)
            total_loss += loss.item()
        return total_loss/len(dataloader)
    def train_conversation_model(self, train_dataloader, dev_dataloader, epoch, is_pretrain=False):
        no_improvement_num = 0
        best_score = 10000000
        if self.rand_sel_ratio == 0.5:
            total_len = len(train_dataloader) //2
        else:
            total_len = len(train_dataloader)
        for e in range(epoch):
            total_loss = 0
            for i, batch in enumerate(train_dataloader):
                loss = self.conversation_training_step(batch)
                total_loss += loss.item()
                if self.rand_sel_ratio == 0.5 and i == total_len:
                    break
            print(f'\t Train. Loss: {total_loss/total_len:.3f}')
            
            '''val_loss = self.evaluate_conversation_model(dev_dataloader)
            print(f'\t Val. Loss: {val_loss:.2f} ')
            print('=' * 50)

            if val_loss < best_score:
                best_score = val_loss
                if is_pretrain:
                    torch.save(self.model.state_dict(), os.path.join(self.dump_path, 'pretrained_model.pth'))
                no_improvement_num = 0
            else:
                no_improvement_num += 1

            if no_improvement_num >= self.early_stop:
                 break'''
        
        return total_loss/total_len
    def train(self, train_dataloader, dev_dataloader, epoch, is_pretrain=False):
        no_improvement_num = 0
        best_score = 0
        if self.rand_sel_ratio == 0.5:
            total_len = len(train_dataloader) //2
        else:
            total_len = len(train_dataloader)
        for e in range(epoch):
            total_loss = 0
            total_acc = 0
            total_h = 0
            for i, (_, X, y) in enumerate(train_dataloader):
                X, y = X.cuda(), y.cuda()
                if self.reward_mode == 'pred_entropy':
                    loss, acc, h = self.training_step(X, y)
                    total_h += h.item()
                else:
                    loss, acc = self.training_step(X, y)
                
                total_loss += loss.item()
                total_acc += acc.item()
                
                if self.rand_sel_ratio == 0.5 and i == total_len:
                    break
            #print(f'\t Train. Loss: {total_loss/total_len:.3f} |  Train. Acc: {total_acc/total_len*100:.2f}% ')
            if self.task == 'ner':
                score_val = self.evaluate_ner(dev_dataloader)
                print(f'\t Val. F1: {score_val:.2f} ')
            elif self.task == 'sentiment':
                val_loss, score_val = self.evaluate_sentiment(dev_dataloader)
                #print(f'\t Val. Loss: {val_loss:.3f}   |  Val. Acc: {score_val*100:.2f}% ')
            #print('=' * 50)

            if score_val > best_score:
                best_score = score_val
                if is_pretrain:
                    torch.save(self.model.state_dict(), os.path.join(self.dump_path, 'pretrained_model.pth'))
                no_improvement_num = 0
            else:
                no_improvement_num += 1

            if no_improvement_num >= self.early_stop:
                 break
        if self.reward_mode == 'pred_entropy':
            return total_acc/total_len*100, score_val, total_h/total_len
        return total_acc/total_len*100, score_val
        


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
        '''From fairseq'''
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if ignore_index is not None:
            pad_mask = target.eq(ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)

        nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
        smooth_loss = smooth_loss.sum()
        eps_i = epsilon / lprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

