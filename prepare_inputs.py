from typing import Any, List, Tuple, Union, Callable
from transformers import BertTokenizerFast
import json
import os
from tqdm import tqdm
import torch
import numpy as np
import transformers

class Instance(object):
    '''
    - piece_ids: L
    - label: 1
    - span: 2
    - feature_path: str
    - sentence_id: str
    - mention_id: str
    '''
    def __init__(self, piece_ids:List[int], label:int, span:Tuple[int, int], feature_path:str, sentence_id:str, mention_id:str) -> None:
        self.piece_ids = piece_ids
        self.label = label
        self.span = span
        self.feature_path = feature_path
        self.sentence_id = sentence_id
        self.mention_id = mention_id

    def todict(self,):
        return {
            "piece_ids": self.piece_ids,
            "label": self.label,
            "span": self.span,
            "feature_path": self.feature_path,
            "sentence_id": self.sentence_id,
            "mention_id": self.mention_id
        }


class MAVENPreprocess(object):

    def __init__(self, root, feature_root, tokenizer, label_start_offset=1, max_length=512, expand_context=False, split_valid=True):
        super().__init__()
        train_file = os.path.join(root, "train.jsonl")
        valid_file = os.path.join(root, "valid.jsonl")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.expand_context = expand_context
        self.label_start_offset = label_start_offset
        self.label_ids = {}
        self.collected = set()
        self.model = None
        self._sentence_buffer = []
        self.feature_root = feature_root
        train_instances = self._file(train_file)
        valid_instances = self._file(valid_file)
        with open("data/MAVEN/train.origin", "wt") as fp:
            for instance in train_instances:
                fp.write(json.dumps(instance.todict())+"\n")

        with open("data/MAVEN/valid.origin", "wt") as fp:
            for instance in valid_instances:
                fp.write(json.dumps(instance.todict())+"\n")

    def add_sentence(self, sentence_id, piece_ids):
        self.collected.add(sentence_id)
        feature_path = os.path.join(self.feature_root, sentence_id)
        if not os.path.exists(f"{feature_path}.npy"):
            self._sentence_buffer.append((piece_ids, sentence_id))
        if len(self._sentence_buffer) >= 128:
            self.clear_sentences()

    def clear_sentences(self,):
        with torch.no_grad():
            if self.model is None:
                self.model = transformers.BertModel.from_pretrained("bert-large-cased").cuda()
                self.model.eval()
            sentences = [t[0] for t in self._sentence_buffer]
            sentence_ids = [t[1] for t in self._sentence_buffer]
            length = [len(t) for t in sentences]
            max_l = max(length)
            masks = torch.FloatTensor([[1] * len(t) + [0] *  (max_l - len(t)) for t in sentences])
            sentences = torch.LongTensor([t + [0] * (max_l - len(t)) for t in sentences])
            outputs = self.model(input_ids=sentences.to(torch.device("cuda")), attention_mask=masks.to(torch.device("cuda")))
            for i, (s_id, s_l) in enumerate(zip(sentence_ids, length)):
                feature_path = os.path.join(self.feature_root, s_id)
                features = outputs[0][i, :s_l, :]
                features = features.cpu().numpy()
                np.save(file=feature_path, arr=features)
        self._sentence_buffer.clear()


    def _file(self, file_path):
        instances = []
        with open(file_path, "rt") as fp:
            for document_line in tqdm(fp):
                document = json.loads(document_line)
                instances.extend(self._document(document))
        self.clear_sentences()
        return instances

    def _document(self, document):
        document_id = document["id"]
        title = document['title']
        sentences = document["content"]
        events = document["events"]
        none_events = document["negative_triggers"]
        instances = []
        for event in events:
            label = self.label_start_offset + event['type_id']
            if event['type'] not in self.label_ids:
                self.label_ids[event['type']] = label
            for mention in event['mention']:
                sentence = sentences[mention['sent_id']]
                sentence_id = f"{document_id}_{mention['sent_id']}"
                span = mention["offset"]
                mention_id = mention["id"]
                piece_ids, span = self._transform_single(
                    token_ids=sentence["tokens"],
                    spans=[span[0],span[0], span[1]-1, span[1]-1],
                    tokenizer=self.tokenizer,
                    is_tokenized=True)
                if len(piece_ids) > 512:
                    print("not none", sentence_id, mention_id, label)
                    continue
                if sentence_id not in self.collected:
                    self.add_sentence(sentence_id, piece_ids)
                span = (span[0], span[3])
                instance = Instance(
                    piece_ids=piece_ids,
                    label=label,
                    span=span,
                    feature_path=f"MAVEN/{sentence_id}",
                    sentence_id=sentence_id,
                    mention_id=mention_id)
                instances.append(instance)
        for mention in none_events:
            sentence = sentences[mention['sent_id']]
            sentence_id = f"{document_id}_{mention['sent_id']}"
            span = mention["offset"]
            mention_id = mention["id"]
            piece_ids, span = self._transform_single(
                token_ids=sentence["tokens"],
                spans=[span[0],span[0], span[1]-1, span[1]-1],
                tokenizer=self.tokenizer,
                is_tokenized=True)
            if len(piece_ids) > 512:
                continue
            if sentence_id not in self.collected:
                self.add_sentence(sentence_id, piece_ids)
            span = (span[0], span[3])
            instance = Instance(
                piece_ids=piece_ids,
                label=0,
                span=span,
                feature_path=f"MAVEN/{sentence_id}",
                sentence_id=sentence_id,
                mention_id=mention_id)
            instances.append(instance)
        return instances


    def _context(self, sentences:List[List[str]]) -> List[Tuple[List[int], int, int]]:
        raise NotImplementedError


    @classmethod
    def _transform_single(cls, token_ids: Union[List[List[str]], List[str], str], spans: Union[List[int], Tuple[int]], tokenizer: BertTokenizerFast, is_tokenized: bool=False) -> Tuple[List[int], List[int]]:
        def _token_span(cls, offsets, s, e):
            ts = []
            i = 0
            while offsets[i][0] <= s:
                i += 1
            ts.append(i - 1)
            i -= 1
            while offsets[i][1] <= e:
                i += 1
            ts.append(i)
            return tuple(ts)
        sent_id = hs = he = ts = te = 0
        _token_ids = _spans = []
        if len(spans) == 4:
            hs, he, ts, te = spans
        else:
            sent_id, hs, he, ts, te = spans
        if isinstance(token_ids, str):
            if is_tokenized:
                raise TypeError("Cannot process single string when 'is_tokenized = True'.")
            else:
                tokens = tokenizer(token_ids, return_offsets_mapping=True)
                _token_ids = tokens["input_ids"]
                offsets = tokens["offset_mapping"][1:-1]
                h = _token_span(offsets, hs, he)
                t = _token_span(offsets, ts, te)
                _spans = [h[0] + 1, h[1] + 1, t[0] + 1, t[1] + 1]
        elif isinstance(token_ids, List):
            if is_tokenized:
                tokens = tokenizer(token_ids, is_pretokenized=True, return_offsets_mapping=True)
                if isinstance(token_ids[0], str):
                    _token_ids = tokens["input_ids"]
                    offsets = tokens["offset_mapping"]
                    token2piece = []
                    piece_idx = 1
                    for x, y in offsets[1:-1]:
                        if x == 0:
                            if len(token2piece) > 0:
                                token2piece[-1].append(piece_idx-1)
                            token2piece.append([piece_idx])
                        piece_idx += 1
                    if len(token2piece[-1]) == 1:
                        token2piece[-1].append(piece_idx-1)
                    _spans = [token2piece[hs][0], token2piece[he][1], token2piece[ts][0], token2piece[te][1]]
                else:
                    token2piece = []
                    piece_idx = 1
                    for x, y in tokens["offset_mapping"][sent_id][1:-1]:
                        if x == 0:
                            if len(token2piece) > 0:
                                token2piece[-1].append(piece_idx-1)
                            token2piece.append([piece_idx])
                        piece_idx += 1
                    if len(token2piece[-1]) == 1:
                        token2piece[-1].append(piece_idx-1)
                    _spans = [token2piece[hs][0], token2piece[he][1], token2piece[ts][0], token2piece[te][1]]
                    _token_ids = []
                    for i, t in enumerate(tokens["input_ids"]):
                        if i == sent_id:
                            _spans = [_t - 1 + len(_token_ids) for _t in _spans]
                        if i > 0:
                            _token_ids.extend(t[1:])
                        else:
                            _token_ids.extend(t)
            else:
                tokens = tokenizer(token_ids, return_offsets_mapping=True)
                if isinstance(token_ids[0], str):
                    offsets = tokens["offset_mapping"][sent_id][1:-1]
                    h = _token_span(offsets, hs, he)
                    t = _token_span(offsets, ts, te)
                    _spans = [h[0], h[1], t[0], t[1]]
                    _token_ids = []
                    for i, t in enumerate(tokens["input_ids"]):
                        if i == sent_id:
                            _spans = [_t + len(_token_ids) for _t in _spans]
                        if i > 0:
                            _token_ids.extend(t[1:])
                        else:
                            _token_ids.extend(t)
                else:
                    raise TypeError("Cannot process list of lists of sentences (list of paragraphs).")

        return _token_ids, _spans


def main():

    MAVEN_PATH = "MAVEN/" # path for original maven dataset
    feature_root = "./data/features"
    bt = BertTokenizerFast.from_pretrained("bert-large-cased")
    m1 = MAVENPreprocess(MAVEN_PATH, feature_root, tokenizer=bt)

if __name__ == "__main__":
    main()