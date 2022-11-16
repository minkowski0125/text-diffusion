import json
import os
import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader
from functools import partial
from ldm.util import get_tokenizer, tokenize, instantiate_from_config
import numpy as np
import torch
from scipy.stats import poisson

def identity_strategy(x):
    input_ids = torch.tensor([t['input_ids'] for t in x], dtype=torch.long)
    attention_mask = torch.tensor([t['attention_mask'] for t in x], dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

class BaseStrategy:
    def __init__(self):
        self.count = 0
    def get_rng(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        rng = np.random.default_rng(self.count * num_workers + worker_id)
        return rng

class BertStrategy(BaseStrategy):
    def __init__(self, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio

    def strategy(self, samples):
        rng = self.get_rng()
        self.count += 1
        diffusion_masks = []
        max_length = samples[0]['attention_mask'].shape[0]
        pos_idx = np.arange(max_length)
        for sample in samples:
            length = (sample['attention_mask']==1).sum()
            diffusion_mask = np.logical_and(rng.random(max_length) < self.mask_ratio, pos_idx < length) * 1
            diffusion_masks.append(diffusion_mask)
        input_ids = torch.tensor([t['input_ids'] for t in samples], dtype=torch.long)
        attention_mask = torch.tensor([t['attention_mask'] for t in samples], dtype=torch.long)
        diffusion_masks = torch.tensor(diffusion_masks, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'diffusion_masks': diffusion_masks
        }

class SpanBertStrategy(BaseStrategy):
    def __init__(self, mask_ratio=0.15, average_block_length=3, max_block_length=40):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.average_block_length = average_block_length
        self.count = 0
        self.block_length_distribution = [poisson.pmf(i, average_block_length) for i in range(1, max_block_length)]
        self.block_length_distribution = np.array(self.block_length_distribution) / sum(self.block_length_distribution)
    
    @staticmethod
    def sample_spans(span_lengths, total_length, rng, offset=0):
        blank_length = total_length - sum(span_lengths)
        m = blank_length - len(span_lengths) + 1
        # places = [rng.randrange(m + 1) for _ in range(len(span_lengths))]
        places = rng.integers(0, m + 1, len(span_lengths))
        places.sort()
        spans = []
        for place, span_length in zip(places, span_lengths):
            start = offset + place
            end = offset + place + span_length
            spans.append((start, end))
            offset += span_length + 1
        return spans
    
    def strategy(self, samples):
        rng = self.get_rng()
        self.count += 1
        diffusion_masks = []
        # max_length = samples[0]['attention_mask'].shape[0]
        # pos_idx = np.arange(max_length)
        for sample in samples:
            length = (sample['attention_mask']==1).sum()
            masked_lengths, masked_count = [], 0
            while masked_count < int(self.mask_ratio * length):
                block_length = rng.choice(range(1, len(self.block_length_distribution) + 1),
                                            p=self.block_length_distribution)
                masked_lengths.append(block_length)
                masked_count += block_length
            rng.shuffle(masked_lengths)
            new_masked_lengths = []
            total_mask = 0
            for l in masked_lengths:
                if total_mask + l <= length:
                    new_masked_lengths.append(l)
                    total_mask += l
                else:
                    break
            spans = self.sample_spans(new_masked_lengths, length, rng)
            # print(spans)
            diffusion_mask = np.zeros_like(sample['attention_mask'])
            for span in spans:
                diffusion_mask[span[0]:span[1]] = 1
            diffusion_masks.append(diffusion_mask)
        input_ids = torch.tensor([t['input_ids'] for t in samples], dtype=torch.long)
        attention_mask = torch.tensor([t['attention_mask'] for t in samples], dtype=torch.long)
        diffusion_masks = torch.tensor(diffusion_masks, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'diffusion_masks': diffusion_masks
        }


class Pile(IterableDataset):
    def __init__(self, root, num_files, sample_length):
        super().__init__()
        self.root = root
        self.num_files = num_files
        self.sample_length = sample_length

    def __iter__(self):
        for i in range(self.num_files):
            pth = os.path.join(self.root, "{:02d}.jsonl".format(i))
            with open(pth, 'r', encoding='utf-8') as f:
                for line in f:
                    dic = json.loads(line)
                    input_ids, attention_mask = tokenize(dic['text'], self.sample_length, return_mask=True)
                    yield {
                            "input_ids": np.array(input_ids, dtype=np.int64),
                            "attention_mask": np.array(attention_mask, dtype=np.int64)
                        } #, 'text': dic['text']}

class IterableDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, collate_fn=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.collate_fn = instantiate_from_config(collate_fn).strategy

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,
                          collate_fn=self.collate_fn)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=self.collate_fn, shuffle=False)

    def _predict_dataloader(self):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=self.collate_fn)
                    
if __name__ == '__main__':
    get_tokenizer('roberta-large')
    pile = Pile('/zhangpai21/data/pile/train', 30, 128)
    strategy = SpanBertStrategy()
    dl = DataLoader(pile, batch_size=2, collate_fn=strategy.strategy)
    for x in dl:
        print(x)
        input()