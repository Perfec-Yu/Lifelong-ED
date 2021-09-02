from typing import Iterable, List, Tuple, Dict, Union
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader#, Sampler
import json
import os

class DataCollection(object):
    datasets = [
        "MAVEN",
        "ACE",
        "ACEE",
        ]
    def __init__(self, root:str, feature_root:str, stream:List[List[Union[str,int]]], splits:Union[List[str],None]=None) -> None:
        self.root = root
        def collect_dataset_split(dataset, split):
            json_f = os.path.join(root, dataset, f"{dataset}.{split}.json")
            jsonl_f = os.path.join(root, dataset, f"{dataset}.{split}.jsonl")
            if os.path.exists(json_f):
                with open(json_f, "rt") as fp:
                    d = json.load(fp)
            else:
                with open(jsonl_f, "rt") as fp:
                    d = [json.loads(t) for t in fp]
            return d
        if splits is None:
            splits = ["train", "dev", "test"]
        for dataset in self.datasets:
            setattr(self, dataset, {
                split: collect_dataset_split(dataset, split) for split in splits
            })
        self.splits = splits
        self.feature_root = feature_root
        self.stream = stream
        self.label2id = {0: 0}
        for task in stream:
            for label in task:
                if label not in self.label2id:
                    self.label2id[label] = len(self.label2id)

    def collect_instance_by_labels(self, labels:Iterable[Union[str, int, Tuple[str, str], Tuple[str, int]]], dataset:Union[str, None]=None) -> Dict[str, List[str]]:
        query = {}
        for label in labels:
            if dataset is None:
                dataset, label = label
            if dataset in query:
                query[dataset].add(label)
            else:
                query[dataset] = {label}
        response = {split: [] for split in self.splits}
        for dataset in query:
            data = getattr(self, dataset, None)
            if data is not None:
                for split in data:
                    response[split].extend([t for t in data[split] if t['label'] in query[dataset]])
        return response

    def feature_path(self, feature_path):
        return os.path.join(self.feature_root, feature_path)


class Instance(object):
    '''
    - piece_ids: L
    - label: 1
    - span: 2
    - feature_path: str
    - sentence_id: str
    - mention_id: str
    '''
    def __init__(self, token_ids:List[int], label:int, span:Tuple[int, int], features:Union[str, torch.FloatTensor], sentence_id:str, mention_id:str) -> None:
        self.token_ids = token_ids
        self.label = label
        self.span = span
        self.features = features
        self.sentence_id = sentence_id
        self.mention_id = mention_id

    def totensor(self,):
        if not isinstance(self.token_ids, torch.LongTensor):
            self.token_ids = torch.LongTensor(self.token_ids)
        if not isinstance(self.span, torch.LongTensor):
            self.span = torch.LongTensor(self.span)
        if not isinstance(self.features, torch.FloatTensor):
            self.features = torch.FloatTensor(self.features)
        return self

    def load_clone(self,):
        if isinstance(self.features, str):
            if not self.features.endswith("npy"):
                self.features += ".npy"
            npy_features = np.load(self.features)
            npy_features = npy_features[self.span, :]
            features = torch.from_numpy(npy_features).float().flatten()
        else:
            features = self.features
        return self.__class__(
            token_ids=self.token_ids,
            label=self.label,
            span=self.span,
            features=features,
            sentence_id=self.sentence_id,
            mention_id=self.mention_id
        )

class Batch(object):

    def __init__(self,
            token_ids: List[torch.LongTensor],
            spans: List[torch.LongTensor],
            labels:List[int],
            features: List[torch.FloatTensor],
            attention_masks:Union[List[torch.FloatTensor], None]=None,
            **kwargs)-> None:
        bsz = len(token_ids)
        assert len(labels) == bsz
        assert len(spans) == bsz
        assert all(len(x) == 2 for x in spans)
        if attention_masks is not None:
            assert len(attention_masks) == bsz
            assert all(len(x) == len(y) for x,y in zip(token_ids, attention_masks))
        _max_length = max(len(x) for x in token_ids)
        self.token_ids = torch.zeros(bsz, _max_length, dtype=torch.long)
        self.attention_masks = torch.zeros(bsz, _max_length, dtype=torch.float)
        for i in range(bsz):
            self.token_ids[i, :token_ids[i].size(0)] = token_ids[i]
            if attention_masks is not None:
                self.attention_masks[i, :token_ids[i].size(0)] = attention_masks[i]
            else:
                self.attention_masks[i, :token_ids[i].size(0)] = 1
        self.spans = torch.stack(spans, dim=0)
        self.labels = torch.LongTensor(labels)
        self.features = torch.stack(features, dim=0)
        self.meta = kwargs

    def pin_memory(self):
        self.token_ids = self.token_ids.pin_memory()
        self.attention_masks = self.attention_masks.pin_memory()
        self.spans = self.spans.pin_memory()
        self.labels = self.labels.pin_memory()
        self.features = self.features.pin_memory()
        return self

    def cuda(self,device:Union[torch.device,int,None]=None):
        assert torch.cuda.is_available()
        self.token_ids = self.token_ids.cuda(device)
        self.attention_masks = self.attention_masks.cuda(device)
        self.spans = self.spans.cuda(device)
        self.labels = self.labels.cuda(device)
        self.features = self.features.cuda(device)
        return self

    def to(self, device:torch.device):
        self.token_ids = self.token_ids.to(device)
        self.attention_masks = self.attention_masks.to(device)
        self.spans = self.spans.to(device)
        self.labels = self.labels.to(device)
        self.features = self.features.to(device)
        return self

    @classmethod
    def from_instances(cls, batch:List[Instance]):
        def slice(attr):
            return [getattr(t, attr) for t in batch]
        batch = [t.totensor() for t in batch]
        return cls(
            token_ids=slice("token_ids"),
            labels=slice("label"),
            features=slice('features'),
            spans=slice("span"),
            sentence_ids=slice("sentence_id"),
            mention_ids=slice("mention_id"))

class LabelDataset(Dataset):
    def __init__(self, instances:List[Instance]) -> None:
        super().__init__()
        instances.sort(key=lambda i:i.label)
        self.label2index = {}
        i = 0
        labels = []
        for instance in instances:
            if len(labels) == 0 or instance.label != labels[-1]:
                if len(labels) > 0:
                    self.label2index[labels[-1]] = (i, len(labels))
                i = len(labels)
            labels.append(instance.label)
        self.label2index[labels[-1]] = (i, len(labels))
        self.instances = instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        instance = self.instances[index]
        return instance.load_clone()

    def get_indices_by_label(self, label:Tuple[str, str]) -> List[Instance]:
        return self.label2index[label]

    def collate_fn(self, batch:List[Instance]) -> Batch:
        return Batch.from_instances(batch)

def get_stage_loaders(root:str,
    feature_root:str,
    batch_size:int,
    streams:List[List[int]],
    num_workers:int=0,
    seed:int=2147483647,
    *args,
    **kwargs):
    dataset_id = 0 if "dataset" not in kwargs else kwargs['dataset']
    def prepare_dataset(instances:List[Dict]) -> List[Instance]:
        instances = [Instance(
            token_ids=instance["piece_ids"],
            span=instance["span"],
            features=collection.feature_path(instance["feature_path"]),
            sentence_id=instance["sentence_id"],
            mention_id=instance["mention_id"],
            label=collection.label2id[instance["label"]]
        ) for instance in instances]
        return instances

    collection = DataCollection(root, feature_root, streams)
    loaders = []
    exclude_none_loaders = []
    all_labels = []
    for stream in streams:
        for t in stream:
            if t not in all_labels:
                all_labels.append(t)
    collect_labels = all_labels.copy()
    for stream in streams:
        stream_instances = collection.collect_instance_by_labels(labels=collect_labels, dataset=collection.datasets[dataset_id])
        for t in stream:
            if t != 0:
                collect_labels.pop(collect_labels.index(t))
        instances_train = prepare_dataset(stream_instances["train"])
        dataset_train = LabelDataset(instances_train)
        train_loader = DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=dataset_train.collate_fn,
            pin_memory=True,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed))
        loaders.append(train_loader)
        exclude_none_labels = [t for t in stream if t != 0]
        exclude_none_instances = collection.collect_instance_by_labels(exclude_none_labels, dataset=collection.datasets[dataset_id])
        exclude_none_instances = prepare_dataset(exclude_none_instances["train"])
        exclude_none_dataset = LabelDataset(exclude_none_instances)
        exclude_none_loader = DataLoader(
            dataset=exclude_none_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=exclude_none_dataset.collate_fn,
            pin_memory=True
            )
        exclude_none_loaders.append(exclude_none_loader)
    labels = list(set([t for stream in streams for t in stream]))
    instances = collection.collect_instance_by_labels(labels, dataset=collection.datasets[dataset_id])
    instances_dev = prepare_dataset(instances["dev"])
    dataset_dev = LabelDataset(instances_dev)
    dev_loader = DataLoader(
        dataset=dataset_dev,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset_dev.collate_fn,
        pin_memory=True
        )
    instances_test = prepare_dataset(instances["test"])
    dataset_test = LabelDataset(instances_test)
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset_test.collate_fn,
        pin_memory=True
        )

    stages = [[collection.label2id[t] for t in stream] for stream in streams]
    return loaders + [dev_loader, test_loader], exclude_none_loaders, stages, collection.label2id

def get_stage_loaders_n(root:str,
    feature_root:str,
    batch_size:int,
    streams:List[List[int]],
    streams_instances:List[List[Dict]],
    num_workers:int=0,
    seed:int=2147483647,
    *args,
    **kwargs):
    dataset_id = 0 if "dataset" not in kwargs else kwargs['dataset']
    def prepare_dataset(instances:List[Dict]) -> List[Instance]:
        instances = [Instance(
            token_ids=instance["piece_ids"],
            span=instance["span"],
            features=collection.feature_path(instance["feature_path"]),
            sentence_id=instance["sentence_id"],
            mention_id=instance["mention_id"],
            label=collection.label2id[instance["label"]]
        ) for instance in instances]
        return instances

    collection = DataCollection(root, feature_root, streams, splits=["dev", "test"])
    loaders = []
    exclude_none_loaders = []
    for stream_instances in streams_instances:
        instances_train = prepare_dataset(stream_instances)
        dataset_train = LabelDataset(instances_train)
        train_loader = DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=dataset_train.collate_fn,
            pin_memory=True,
            num_workers=num_workers,
            generator=torch.Generator().manual_seed(seed))
        loaders.append(train_loader)
        exclude_none_instances = [t for t in stream_instances if t['label'] != 0]
        exclude_none_instances = prepare_dataset(exclude_none_instances)
        exclude_none_dataset = LabelDataset(exclude_none_instances)
        exclude_none_loader = DataLoader(
            dataset=exclude_none_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=exclude_none_dataset.collate_fn,
            pin_memory=True
            )
        exclude_none_loaders.append(exclude_none_loader)
    labels = list(set([t for stream in streams for t in stream]))
    instances = collection.collect_instance_by_labels(labels, dataset=collection.datasets[dataset_id])
    instances_dev = prepare_dataset(instances["dev"])
    dataset_dev = LabelDataset(instances_dev)
    dev_loader = DataLoader(
        dataset=dataset_dev,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset_dev.collate_fn,
        pin_memory=True,
        num_workers=4
        )
    instances_test = prepare_dataset(instances["test"])
    dataset_test = LabelDataset(instances_test)
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset_test.collate_fn,
        pin_memory=True,
        num_workers=4
        )

    stages = [[collection.label2id[t] for t in stream] for stream in streams]
    return loaders + [dev_loader, test_loader], exclude_none_loaders, stages, collection.label2id

def test():
    l = get_stage_loaders(root="./data/", feature_root="/scratch/pengfei4/LInEx/data", batch_size=2, num_steps=5, episode_num_classes=4, episode_num_instances=3, episode_num_novel_classes=2, evaluation_num_instances=6)

if __name__ == "__main__":
    test()
