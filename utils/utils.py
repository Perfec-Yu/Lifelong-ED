import json
from typing import Union, List
def get_label2id(streams:Union[List[int], None]=None, stream_file:Union[str, None]=None):
    if streams is None:
        with open(stream_file, "rt") as sf:
            streams = json.load(sf)
    label2id = {0: 0}
    for task in streams:
        for label in task:
            if label not in label2id:
                label2id[label] = len(label2id)
    return label2id