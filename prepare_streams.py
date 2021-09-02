import json
from collections import Counter
import numpy as np
nstreams = 5
seed = 2227341903
dataset ="ACE"

np.random.seed(seed)

def ninstances(stream, label_freqs):
    return sum([label_freqs[t] for t in stream])

with open(f"data/{dataset}/{dataset}.train.jsonl") as fp:
    x = [json.loads(t) for t in fp]
streams = [[] for _ in range(nstreams)]
label_freqs = dict(Counter([t['label'] for t in x]).most_common())
labels = list(label_freqs.keys())
labels_shuffle = [labels[i] for i in np.random.permutation(np.arange(len(labels)))]
for l in labels_shuffle:
    l = int(l)
    if l == 0:
        continue
    least_stream = 0
    least_instances = ninstances(streams[0], label_freqs)
    for i in range(1, len(streams)):
        cinstances = ninstances(streams[i], label_freqs)
        if cinstances < least_instances:
            least_stream = i
            least_instances = cinstances
    streams[least_stream].append(l)
print("Without None", [(ninstances(t, label_freqs), len(t)) for t in streams])
for t in streams:
    t.append(0)
print("With None", [ninstances(t, label_freqs) for t in streams])
json.dump(obj=streams, fp=open(f"data/{dataset}/streams.json", "wt"))