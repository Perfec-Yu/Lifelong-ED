import json
import os
import numpy as np
from tqdm import tqdm
from utils.datastream import DataCollection

def split_stream_instance(root, feature_root, streams, dataset_id):
    collection = DataCollection(root, feature_root, streams)
    np.random.seed(2147483647)
    none_instances = collection.collect_instance_by_labels(labels=[0], dataset=collection.datasets[dataset_id])
    train_none = none_instances["train"]
    rest_instances = []
    stream_instances = []
    nstreams = len(streams)
    collected = set()
    for stream in streams:
        stream_instances.append([])
        rest_instances.append([])
        for label in tqdm(stream):
            if label == 0:
                continue
            if label not in collected:
                collected.add(label)
            else:
                print(label)
            label_instances = collection.collect_instance_by_labels(labels=[label], dataset=collection.datasets[dataset_id])
            train_instances = label_instances["train"]
            for t in train_instances:
                if t['label'] != label:
                    print(label, t, stream)
                    input()
            rand_perm = np.random.permutation(len(train_instances))
            stream_instances[-1].extend([train_instances[i] for i in range(int(len(rand_perm)*0.9))])
            rest_instances[-1].extend([train_instances[i] for i in range(int(len(rand_perm)*0.9), len(rand_perm))])
            rest = iter(train_instances[i] for i in range(int(len(rand_perm)*0.9), len(rand_perm)))
            for instance in rest:
                instance["original_label"] = instance["label"] + 0
                instance["label"] = 0
                assert instance['label'] == 0, instance["original_label"] == label
    
    for i in range(len(streams)):
        nrest = len(rest_instances[i])
        drest = nrest // (nstreams - 1)
        irest = drest
        for j in range(nstreams):
            if j != i:
                stream_instances[j].extend(rest_instances[i][irest-drest:irest])
                irest += drest
                if irest > nrest:
                    stream_instances[j].extend(rest_instances[i][irest-drest:irest])
    
    rand_perm = np.random.permutation(len(train_none))
    nnone = len(train_none)
    dnone = nnone // nstreams
    inone = dnone
    train_none = [train_none[i] for i in rand_perm]
    for j in range(nstreams):
        stream_instances[j].extend(train_none[inone-dnone:inone])
        print(j, len(stream_instances[j]))
        inone += dnone
        if inone > nnone:
            stream_instances[j].extend(train_none[inone-dnone:inone])
    return stream_instances

if __name__ == "__main__":
    root="./data/"
    feature_root="./data/features"
    dataset_id = 0
    streams = json.load(open(os.path.join(root, "MAVEN", "streams.json")))
    instances = split_stream_instance(root, feature_root, streams, dataset_id)
    json.dump(instances, open(os.path.join(root, "MAVEN", "stream_instances.json"), "wt"), indent=4)
