import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import math
from typing import Any, Dict, Tuple, List, Union, Set
import warnings
from collections import OrderedDict
from torch.nn.modules.linear import Linear
from torchmeta.modules import MetaLinear, MetaSequential, MetaModule
from tqdm import tqdm

class LInEx(MetaModule):
    def __init__(self,input_dim:int,hidden_dim:int,max_slots:int,init_slots:int,device:Union[torch.device, None]=None,**kwargs)->None:
        super().__init__()
        if input_dim != hidden_dim:
            self.input_map = MetaSequential(OrderedDict({
                "linear_0": MetaLinear(input_dim, hidden_dim),
                "relu_0": nn.ReLU(),
                "dropout_0": nn.Dropout(0.2),
                "linear_1": MetaLinear(hidden_dim, hidden_dim),
                "relu_1": nn.ReLU()
                }))
        else:
            self.input_map = lambda x:x
        self.classes = MetaLinear(hidden_dim, max_slots, bias=False)
        _mask = torch.zeros(1, max_slots, dtype=torch.float, device=device)
        _mask[:, init_slots:] = float("-inf")
        self.register_buffer(name="_mask", tensor=_mask)
        self.crit = nn.CrossEntropyLoss()
        self.device = device
        self.to(device=device)
        self.nslots = init_slots
        self.max_slots = max_slots
        self.maml = True
        self.outputs = {}
        self.history = None
        self.exemplar_features = None
        self.exemplar_labels = None
        self.dev_exemplar_features = None
        self.dev_exemplar_labels = None

    @property
    def mask(self,):
        self._mask[:, :self.nslots] = 0
        self._mask[:, self.nslots:] = float("-inf")
        return self._mask

    def idx_mask(self, idx:Union[torch.LongTensor, int, List[int], None]=None, max_idx:Union[torch.LongTensor, int, None]=None):
        assert (idx is not None) or (max_idx is not None)
        assert (idx is None) or (max_idx is None)
        mask = torch.zeros_like(self._mask) + float("-inf")
        if idx is not None:
            mask[:, idx] = 0
        if max_idx is not None:
            if isinstance(max_idx, torch.LongTensor):
                max_idx = max_idx.item()
            mask[:, :max_idx] = 0
        return mask

    @property
    def features(self):
        return self.classes.weight[:self.nslots]

    def forward(self, batch, nslots:int=-1, exemplar:bool=False, exemplar_distill:bool=False, feature_distill:bool=False, mul_distill=False, distill:bool=False, return_loss:bool=True, return_feature:bool=False, tau:float=1.0, log_outputs:bool=True, params=None):
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            features, labels = batch
        else:
            features, labels = batch.features, batch.labels
        inputs = self.input_map(features, params=self.get_subdict(params, "input_map"))
        scores = self.classes(inputs, params=self.get_subdict(params, "classes"))
        if torch.any(torch.isnan(scores)):
            print(scores[0])
            input('a')
        if nslots == -1:
            scores += self.mask
            if torch.any(torch.isnan(scores)):
                print(scores[0])
                input()
            nslots = self.nslots
        else:
            scores += self.idx_mask(max_idx=nslots)
        scores[:, 0] = 0
        if scores.size(0) != labels.size(0):
            assert scores.size(0) % labels.size(0) == 0
            labels = labels.repeat_interleave(scores.size(0) // labels.size(0), dim=0)
        else:
            labels = labels
        if log_outputs:
            pred = torch.argmax(scores, dim=1)
            acc = torch.mean((pred == labels).float())
            self.outputs["accuracy"] = acc.item()
            self.outputs["prediction"] = pred.detach().cpu()
            self.outputs["label"] = labels.detach().cpu()
            self.outputs["input_features"] = features.detach().cpu()
            self.outputs["encoded_features"] = inputs.detach().cpu()
        if return_loss:
            labels.masked_fill_(labels >= nslots, 0)
            valid = labels < nslots
            nvalid = torch.sum(valid.float())
            if nvalid == 0:
                loss = 0
            else:
                loss = self.crit(scores[valid], labels[valid])
                if torch.isnan(loss):
                    print(labels, nslots, scores[:, :nslots])
                    input()
            if distill and self.history is not None:
                old_scores, old_inputs = self.forward(batch, nslots=self.history["nslots"], return_loss=False, log_outputs=False, return_feature=True, params=self.history["params"])
                old_scores = old_scores.detach()
                old_inputs = old_inputs.detach()
                new_scores = scores[:, :self.history["nslots"]]
                if mul_distill:
                    loss_distill = - torch.sum(torch.softmax(old_scores*tau, dim=1) * torch.log_softmax(new_scores*tau, dim=1), dim=1).mean()
                    old_dist = torch.softmax(old_scores/tau, dim=1)
                    old_valid = (old_dist[:, 0] < 0.9)
                    old_num = torch.sum(old_valid.float())
                    if old_num > 0:
                        # print(old_dist[old_valid].topk(5, dim=1), batch.labels[old_valid])
                        # input()
                        loss_mul_distill = - torch.sum(old_dist[old_valid] * torch.log_softmax(new_scores[old_valid], dim=1), dim=1).sum()
                        loss_distill = (loss_distill * old_dist.size(0) + loss_mul_distill) / (old_dist.size(0) + old_num)
                else:
                    loss_distill = - torch.sum(torch.softmax(old_scores*tau, dim=1) * torch.log_softmax(new_scores*tau, dim=1), dim=1).mean()
                if feature_distill:
                    loss_f_distill = (1 - (old_inputs / old_inputs.norm(dim=-1, keepdim=True) * inputs / inputs.norm(dim=-1, keepdim=True)).sum(dim=-1)).mean(dim=0)
                    loss_distill += loss_f_distill

                d_weight = self.history["nslots"]
                c_weight = (self.nslots - self.history["nslots"])
                loss = ( d_weight * loss_distill+ c_weight* loss) / (d_weight+c_weight)
                if torch.isnan(loss):
                    print(old_scores, new_scores)
                    input()
            if exemplar and self.exemplar_features is not None:
                if self.exemplar_features.size(0) < 128:
                    exemplar_inputs = self.input_map(self.exemplar_features.to(self.device), params=self.get_subdict(params, "input_map"))
                    exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                else:
                    exemplar_scores = []
                    exemplar_inputs = []
                    for _beg in range(0, self.exemplar_features.size(0), 128):
                        _features = self.exemplar_features[_beg:_beg+128, :]
                        _inputs = self.input_map(_features.to(self.device), params=self.get_subdict(params, "input_map"))
                        exemplar_inputs.append(_inputs)
                        exemplar_scores.append(self.classes(_inputs, params=self.get_subdict(params, "classes")))
                    exemplar_inputs = torch.cat(exemplar_inputs, dim=0)
                    exemplar_scores = torch.cat(exemplar_scores, dim=0)
                exemplar_scores[:, 0] = 0.
                loss_exemplar = self.crit(exemplar_scores+self.mask, self.exemplar_labels.to(self.device))
                if torch.isnan(loss_exemplar):
                    print(self.exemplar_labels, nslots)
                    input()
                if exemplar_distill:
                    if self.exemplar_features.size(0) < 128:
                        exemplar_old_inputs = self.input_map(self.exemplar_features.to(self.device), params=self.get_subdict(self.history["params"], "input_map"))
                        exemplar_old_scores = self.classes(exemplar_old_inputs, params=self.get_subdict(self.history["params"], "classes"))
                    else:
                        exemplar_old_scores = []
                        exemplar_old_inputs = []
                        for _beg in range(0, self.exemplar_features.size(0), 128):
                            _features = self.exemplar_features[_beg:_beg+128, :]
                            _inputs = self.input_map(_features.to(self.device), params=self.get_subdict(self.history["params"], "input_map"))
                            exemplar_old_inputs.append(_inputs)
                            exemplar_old_scores.append(self.classes(_inputs, params=self.get_subdict(self.history["params"], "classes")))
                        exemplar_old_inputs = torch.cat(exemplar_old_inputs, dim=0)
                        exemplar_old_scores = torch.cat(exemplar_old_scores, dim=0)
                    exemplar_old_scores[:, 0] = 0.
                    exemplar_old_scores = exemplar_old_scores[:self.history["nslots"]]
                    loss_exemplar_distill = - torch.sum(torch.softmax(exemplar_old_scores[:self.history["nslots"]]*tau, dim=1) * torch.log_softmax(exemplar_scores[:self.history["nslots"]], dim=1), dim=1).mean()
                    if feature_distill:
                        loss_exemplar_feat_distill = (1 - (exemplar_old_inputs / exemplar_old_inputs.norm(dim=-1, keepdim=True) * exemplar_inputs / exemplar_inputs.norm(dim=-1, keepdim=True)).sum(dim=-1)).mean(dim=0)
                        loss_exemplar_distill += loss_exemplar_feat_distill
                    d_weight = self.history["nslots"]
                    c_weight = (self.nslots - self.history["nslots"])
                    loss_exemplar = (d_weight * loss_exemplar_distill+ c_weight* loss_exemplar) / (d_weight+c_weight)
                e_weight = self.exemplar_features.size(0)
                loss = (nvalid * loss + e_weight * loss_exemplar) / (nvalid + e_weight)
                if torch.isnan(loss):
                    print(loss, loss_exemplar)
            return loss
        else:
            if return_feature:
                return scores[:, :nslots], inputs
            else:
                return scores[:, :nslots]

    def score(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def clone_params(self,):
        return OrderedDict({k:v.clone().detach() for k,v in self.meta_named_parameters()})

    def set_history(self,):
        self.history = {"params": self.clone_params(), "nslots": self.nslots}

    def set_exemplar(self, dataloader, q:int=20, params=None, label_sets:Union[List, Set, None]=None, collect_none:bool=False, use_input:bool=False, output_only:bool=False, output:Union[str, None]=None):
        self.eval()
        with torch.no_grad():
            ifeat = []; ofeat = []; label = []
            num_batches = len(dataloader)
            for batch in tqdm(dataloader, "collecting exemplar", ncols=128):
                batch = batch.to(self.device)
                loss = self.forward(batch, params=params)
                ifeat.append(self.outputs["input_features"])
                if use_input:
                    ofeat.append(self.outputs["input_features"])
                else:
                    ofeat.append(self.outputs["encoded_features"])
                label.append(self.outputs["label"])
            ifeat = torch.cat(ifeat, dim=0)
            ofeat = torch.cat(ofeat, dim=0)
            label = torch.cat(label, dim=0)
            nslots = max(self.nslots, torch.max(label).item()+1)
            exemplar = {}
            if label_sets is None:
                if collect_none:
                    label_sets = range(nslots)
                else:
                    label_sets = range(1, nslots)
            else:
                if collect_none:
                    if 0 not in label_sets:
                        label_sets = sorted([0] + list(label_sets))
                    else:
                        label_sets = sorted(list(label_sets))
                else:
                    label_sets = sorted([t for t in label_sets if t != 0])
            for i in label_sets:
                idx = (label == i)
                if i == 0:
                    # random sample for none type
                    nidx = torch.nonzero(idx, as_tuple=True)[0].tolist()
                    exemplar[i] = numpy.random.choice(nidx, q, replace=False).tolist()
                    continue
                if torch.any(idx):
                    exemplar[i] = []
                    nidx = torch.nonzero(idx, as_tuple=True)[0].tolist()
                    mfeat = torch.mean(ofeat[idx], dim=0, keepdims=True)
                    if len(nidx) < q:
                        exemplar[i].extend(nidx * (q // len(nidx)) + nidx[:(q % len(nidx))])
                    else:
                        for j in range(q):
                            if j == 0:
                                dfeat = torch.sum((ofeat[nidx] - mfeat)**2, dim=1)
                            else:
                                cfeat = ofeat[exemplar[i]].sum(dim=0, keepdims=True)
                                cnum = len(exemplar[i])
                                dfeat = torch.sum((mfeat * (cnum + 1) - ofeat[nidx] - cfeat)**2, )
                            tfeat = torch.argmin(dfeat)
                            exemplar[i].append(nidx[tfeat])
                            nidx.pop(tfeat.item())
            exemplar = {i: ifeat[v] for i,v in exemplar.items()}
            exemplar_features = []
            exemplar_labels = []
            for label, features in exemplar.items():
                exemplar_features.append(features)
                exemplar_labels.extend([label]*features.size(0))
            exemplar_features = torch.cat(exemplar_features, dim=0).cpu()
            exemplar_labels = torch.LongTensor(exemplar_labels).cpu()
            if not output_only or output is not None:
                if output == "train" or output is None:
                    if self.exemplar_features is None:
                        self.exemplar_features = exemplar_features
                        self.exemplar_labels = exemplar_labels
                    else:
                        self.exemplar_features = torch.cat((self.exemplar_features, exemplar_features), dim=0)
                        self.exemplar_labels = torch.cat((self.exemplar_labels, exemplar_labels), dim=0)
                elif output == "dev":
                    if self.dev_exemplar_features is None:
                        self.dev_exemplar_features = exemplar_features
                        self.dev_exemplar_labels = exemplar_labels
                    else:
                        self.dev_exemplar_features = torch.cat((self.dev_exemplar_features, exemplar_features), dim=0)
                        self.dev_exemplar_labels = torch.cat((self.dev_exemplar_labels, exemplar_labels), dim=0)

        return {i: v.cpu() for i,v in exemplar.items()}

    def initialize(self, exemplar, ninstances:Dict[int, int], gamma:float=1.0, tau:float=1.0, alpha:float=0.5, params=None):
        self.eval()
                
        with torch.no_grad():
            weight_norm = torch.norm(self.classes.weight[1:self.nslots], dim=1).mean(dim=0)
            label_inits = []
            label_kt = {}
            for label, feats in exemplar.items():
                exemplar_inputs = self.input_map(feats.to(self.device), params=self.get_subdict(params, "input_map"))
                exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                exemplar_scores = exemplar_scores + self.mask
                exemplar_scores[:, 0] = 0
                exemplar_weights = torch.softmax(exemplar_scores * tau, dim=1)
                normalized_inputs = exemplar_inputs / torch.norm(exemplar_inputs, dim=1, keepdim=True) * weight_norm
                proto = (exemplar_weights[:, :1] * normalized_inputs).mean(dim=0)
                knowledge = torch.matmul(exemplar_weights[:, 1:self.nslots], self.classes.weight[1:self.nslots]).mean(dim=0)
                gate = alpha * math.exp(- ninstances[label] * gamma)
                # gate = 1 / (1 + ninstances[label] * gamma)
                rnd = torch.randn_like(proto) * weight_norm / math.sqrt(self.classes.weight.size(1))
                initvec = proto *  gate + knowledge * gate + (1 - gate) * rnd
                label_inits.append((label, initvec.cpu()))
                label_kt[label] = exemplar_weights.mean(dim=0).cpu()
            label_inits.sort(key=lambda t:t[0])
            inits = []
            for i, (label, init) in enumerate(label_inits):
                assert label == self.nslots + i
                inits.append(init)
            inits = torch.stack(inits, dim=0)
            self.outputs["new2old"] = label_kt
        return inits.detach()
    
    def initialize2(self, exemplar, ninstances:Dict[int, int], gamma:float=1.0, tau:float=1.0, alpha:float=0.5, delta:float=0.5, params=None):
        self.eval()
        def top_p(probs, p=0.9):
            _val, _idx = torch.sort(probs, descending=True, dim=1)
            top_mask = torch.zeros_like(probs).float() - float("inf")
            for _type in range(probs.size(0)):
                accumulated = 0
                _n = 0
                while accumulated < p or _n <= 1:
                    top_mask[_type, _idx[_type, _n]] = 0
                    accumulated += _val[_type, _n]
                    _n += 1
            return top_mask
        with torch.no_grad():
            weight_norm = torch.norm(self.classes.weight[1:self.nslots], dim=1).mean(dim=0)
            label_inits = []
            label_kt = {}
            for label, feats in exemplar.items():
                exemplar_inputs = self.input_map(feats.to(self.device), params=self.get_subdict(params, "input_map"))
                exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                exemplar_scores = exemplar_scores + self.mask
                exemplar_scores[:, 0] = 0
                top_mask = top_p(torch.softmax(exemplar_scores, dim=1))
                exemplar_scores = exemplar_scores + top_mask
                exemplar_scores[:, 0] = 0
                exemplar_weights = torch.softmax(exemplar_scores * tau, dim=1)
                normalized_inputs = exemplar_inputs / torch.norm(exemplar_inputs, dim=1, keepdim=True) * weight_norm
                proto = delta * (exemplar_weights[:, :1] * normalized_inputs).mean(dim=0)
                kweight = (1 - exemplar_weights[:, :1])
                knowledge = torch.matmul((1-delta*exemplar_weights[:, :1]) * (exemplar_weights[:, 1:self.nslots] + 1e-8) / torch.clamp(1 - exemplar_weights[:, :1], 1e-8), self.classes.weight[1:self.nslots]).mean(dim=0)
                gate = alpha * math.exp(- ninstances[label] * gamma)
                rnd = torch.randn_like(proto) * weight_norm / math.sqrt(self.classes.weight.size(1))
                initvec = proto *  gate + knowledge * gate + (1 - gate) * rnd
                if torch.any(torch.isnan(initvec)):
                    print(proto, knowledge, rnd, gate, exemplar_weights[:, :1], exemplar_scores[-1, :self.nslots])
                    input()
                label_inits.append((label, initvec.cpu()))
                label_kt[label] = exemplar_weights.mean(dim=0).cpu()
            label_inits.sort(key=lambda t:t[0])
            inits = []
            for i, (label, init) in enumerate(label_inits):
                assert label == self.nslots + i
                inits.append(init)
            inits = torch.stack(inits, dim=0)
            self.outputs["new2old"] = label_kt
        return inits.detach()
 
    def set(self, features:torch.tensor, ids:Union[int, torch.Tensor, List, None]=None, max_id:int=-1):
        with torch.no_grad():
            if isinstance(ids, (torch.Tensor, list)):
                if torch.any(ids > self.nslots):
                    warnings.warn("Setting features to new classes. Using 'extend' or 'append' is preferred for new classes")
                self.classes.weight[ids] = features
            elif isinstance(ids, int):
                self.classes.weight[ids] = features
            else:
                if max_id == -1:
                    raise ValueError(f"Need input for either ids or max_id")
                self.classes.weight[:max_id] = features

    def append(self, feature):
        with torch.no_grad():
            self.classes.weight[self.nslots] = feature
            self.nslots += 1

    def extend(self, features):
        with torch.no_grad():
            features = features.to(self.device)
            if len(features.size()) == 1:
                warnings.warn("Extending 1-dim feature vector. Using 'append' instead is preferred.")
                self.append(features)
            else:
                nclasses = features.size(0)
                self.classes.weight[self.nslots:self.nslots+nclasses] = features
                self.nslots += nclasses

class BIC(LInEx):
    def __init__(self,input_dim:int,hidden_dim:int,max_slots:int,init_slots:int,device:Union[torch.device, None]=None, **kwargs)->None:
        super().__init__(input_dim,hidden_dim,max_slots,init_slots,device,**kwargs)
        self.correction_weight = nn.Parameter(torch.ones(1, dtype=torch.float, device=self.device, requires_grad=True))
        self.correction_bias = nn.Parameter(torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=True))
        self.correction_stream = [init_slots]

    def add_stream(self, num_classes):
        self.correction_stream.append(self.correction_stream[-1]+num_classes)

    def forward(self, batch, nslots:int=-1, bias_correction:str="none", exemplar:bool=False, exemplar_distill:bool=False, distill:bool=False, return_loss:bool=True, tau:float=1.0, log_outputs:bool=True, params=None):
        assert bias_correction in ["none", "last", "current"]
        if distill:
            assert bias_correction != "current"
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            features, labels = batch
        else:
            features, labels = batch.features, batch.labels
        inputs = self.input_map(features, params=self.get_subdict(params, "input_map"))
        scores = self.classes(inputs, params=self.get_subdict(params, "classes"))
        if nslots == -1:
            scores += self.mask
            nslots = self.nslots
        else:
            scores += self.idx_mask(max_idx=nslots)
        scores[:, 0] = 0
        if bias_correction == "current":
            assert len(self.correction_stream) >= 2
            scores[:, self.correction_stream[-2]:self.correction_stream[-1]] *= self.correction_weight
            scores[:, self.correction_stream[-2]:self.correction_stream[-1]] += self.correction_bias
        if scores.size(0) != labels.size(0):
            assert scores.size(0) % labels.size(0) == 0
            labels = labels.repeat_interleave(scores.size(0) // labels.size(0), dim=0)
        else:
            labels = labels
        if log_outputs:
            pred = torch.argmax(scores, dim=1)
            acc = torch.mean((pred == labels).float())
            self.outputs["accuracy"] = acc.item()
            self.outputs["prediction"] = pred.detach().cpu()
            self.outputs["label"] = labels.detach().cpu()
            self.outputs["input_features"] = features.detach().cpu()
            self.outputs["encoded_features"] = inputs.detach().cpu()
        if return_loss:
            labels.masked_fill_(labels >= nslots, 0)
            valid = labels < nslots
            nvalid = torch.sum(valid.float())
            if nvalid == 0:
                loss = 0
            else:
                loss = self.crit(scores[valid], labels[valid])
            if distill and self.history is not None:
                old_scores = self.forward(batch, nslots=self.history["nslots"], return_loss=False, log_outputs=False, params=self.history["params"]).detach()
                if bias_correction == "last":
                    old_scores[:, self.correction_stream[-2]:self.correction_stream[-1]] *= self.history['correction_weight']
                    old_scores[:, self.correction_stream[-2]:self.correction_stream[-1]] += self.history['correction_bias']
                new_scores = scores[:, :self.history["nslots"]]
                loss_distill = - torch.sum(torch.softmax(old_scores*tau, dim=1) * torch.log_softmax(new_scores*tau, dim=1), dim=1).mean()
                d_weight = self.history["nslots"]
                c_weight = (self.nslots - self.history["nslots"])
                loss = ( d_weight * loss_distill+ c_weight* loss) / (d_weight+c_weight)
            if exemplar and self.exemplar_features is not None:
                if self.exemplar_features.size(0) < 128:
                    exemplar_inputs = self.input_map(self.exemplar_features.to(self.device), params=self.get_subdict(params, "input_map"))
                    exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                else:
                    exemplar_scores = []
                    for _beg in range(0, self.exemplar_features.size(0), 128):
                        _features = self.exemplar_features[_beg:_beg+128, :]
                        _inputs = self.input_map(_features.to(self.device), params=self.get_subdict(params, "input_map"))
                        exemplar_scores.append(self.classes(_inputs, params=self.get_subdict(params, "classes")))
                    exemplar_scores = torch.cat(exemplar_scores, dim=0)
                exemplar_scores[:, 0] = 0.
                loss_exemplar = self.crit(exemplar_scores+self.mask, self.exemplar_labels.to(self.device))
                if exemplar_distill:
                    if self.exemplar_features.size(0) < 128:
                        exemplar_old_inputs = self.input_map(self.exemplar_features.to(self.device), params=self.get_subdict(self.history["params"], "input_map"))
                        exemplar_old_scores = self.classes(exemplar_old_inputs, params=self.get_subdict(self.history["params"], "classes"))
                    else:
                        exemplar_old_scores = []
                        for _beg in range(0, self.exemplar_features.size(0), 128):
                            _features = self.exemplar_features[_beg:_beg+128, :]
                            _inputs = self.input_map(_features.to(self.device), params=self.get_subdict(self.history["params"], "input_map"))
                            exemplar_old_scores.append(self.classes(_inputs, params=self.get_subdict(self.history["params"], "classes")))
                        exemplar_old_scores = torch.cat(exemplar_old_scores, dim=0)
                    exemplar_old_scores[:, 0] = 0.
                    if bias_correction == "last":
                        exemplar_old_scores[:, self.correction_stream[-2]:self.correction_stream[-1]] *= self.history['correction_weight']
                        exemplar_old_scores[:, self.correction_stream[-2]:self.correction_stream[-1]] += self.history['correction_bias']
                    exemplar_old_scores = exemplar_old_scores[:self.history["nslots"]]
                    loss_exemplar_distill = - torch.sum(torch.softmax(exemplar_old_scores[:self.history["nslots"]]*tau, dim=1) * torch.log_softmax(exemplar_scores[:self.history["nslots"]], dim=1), dim=1).mean()
                    d_weight = self.history["nslots"]
                    c_weight = (self.nslots - self.history["nslots"])
                    loss_exemplar = (d_weight * loss_exemplar_distill+ c_weight* loss_exemplar) / (d_weight+c_weight)
                e_weight = self.exemplar_features.size(0)
                loss = (nvalid * loss + e_weight * loss_exemplar) / (nvalid + e_weight)
                if torch.isnan(loss):
                    print(loss, loss_exemplar)
            return loss
        else:
            return scores[:, :nslots]

    def forward_correction(self, *args, **kwargs):
        '''
        training:
            entropy: normal
            distill:
                old, last
                Fold, Fold * correction_weight + correction_bias,
        '''
        if len(args) >= 3:
            args[2] = "current"
        else:
            kwargs["bias_correction"] = "current"
        return self.forward(*args,**kwargs)

    def set_history(self):
        super().set_history()
        self.history["correction_weight"] = self.correction_weight.item()
        self.history["correction_bias"] = self.correction_bias.item()

    def score(self, *args, **kwargs):
        if len(self.correction_stream) >= 2:
            return self.forward_correction(*args, **kwargs)
        else:
            if len(args) >= 3:
                args[2] = "none"
            else:
                kwargs["bias_correction"] = "none"
            return self.forward(*args, **kwargs)

class ICARL(LInEx):
    def __init__(self,input_dim:int,hidden_dim:int,max_slots:int,init_slots:int,device:Union[torch.device, None]=None, **kwargs)->None:
        super().__init__(input_dim,hidden_dim,max_slots,init_slots,device,**kwargs)
        self.none_feat = None
    
    def set_none_feat(self, dataloader, params=None):
        self.eval()
        with torch.no_grad():
            ifeat = []; ofeat = []; label = []
            num_batches = len(dataloader)
            for batch in tqdm(dataloader, "collecting exemplar"):
                batch = batch.to(self.device)
                loss = self.forward(batch, params=params)
                ifeat.append(self.outputs["input_features"])
                ofeat.append(self.outputs["encoded_features"])
                label.append(self.outputs["label"])
            ifeat = torch.cat(ifeat, dim=0)
            ofeat = torch.cat(ofeat, dim=0)
            label = torch.cat(label, dim=0)
            nslots = max(self.nslots, torch.max(label).item()+1)
            exemplar = {}
            idx = (label == 0)
            self.none_feat = ofeat[idx].mean(dim=0).cpu()
            return self.none_feat

    def score(self, batch, exemplar=None, params=None):
        if exemplar is None:
            exemplar_labels, exemplar_features = self.exemplar_labels, self.exemplar_features
        else:
            exemplar_labels, exemplar_features = exemplar

        inputs = self.input_map(batch.features, params=self.get_subdict(params, "input_map"))
        scores = []
        scores.append(- torch.sum((inputs - self.none_feat.to(inputs.device).unsqueeze(0))**2, dim=1))
        for i in range(1, self.nslots):
            label_idx = (exemplar_labels == i)
            label_features = exemplar_features[label_idx]
            label_inputs = self.input_map(label_features.to(inputs.device), params=self.get_subdict(params, "input_map")).mean(dim=0, keepdim=True)
            scores.append(- torch.sum((inputs - label_inputs)**2, dim=1))
        scores = torch.stack(scores, dim=0).transpose(0, 1)
        labels = batch.labels
        if scores.size(0) != labels.size(0):
            assert scores.size(0) % labels.size(0) == 0
            labels = labels.repeat_interleave(scores.size(0) // labels.size(0), dim=0)
        pred = torch.argmax(scores, dim=1)
        acc = torch.mean((pred == labels).float())
        labels.masked_fill_(labels >= self.nslots, 0)
        valid = labels < self.nslots
        nvalid = torch.sum(valid.float())
        if nvalid == 0:
            loss = 0
        else:
            loss = self.crit(scores[valid], labels[valid])
        self.outputs["accuracy"] = acc.item()
        self.outputs["prediction"] = pred.detach().cpu()
        self.outputs["label"] = labels.detach().cpu()
        self.outputs["input_features"] = batch.features.detach().cpu()
        self.outputs["encoded_features"] = inputs.detach().cpu()
        return loss

def test(): # sanity check
    m = LInEx(nhead=8,nlayers=3,hidden_dim=512,input_dim=2048,max_slots=30,init_slots=9,device=torch.device("cpu"))

if __name__ == "__main__":
    test()
