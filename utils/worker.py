import torch
import os
import logging
import datetime
import numpy as np
import torch
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Union

class Record(object):
    def __init__(self, percentage=False):
        super().__init__()
        self.value = 0.
        self.num = 0.
        self.percentage = percentage

    def __iadd__(self, val):
        self.value += val
        self.num += 1
        return self

    def reset(self):
        self.value = 0.
        self.num = 0.

    def __str__(self):
        if self.percentage:
            display = f"{self.value / max(1, self.num) * 100:.2f}%"
        else:
            display = f"{self.value / max(1, self.num):.4f}"
        return display

    @property
    def true_value(self,):
        return self.value / max(1, self.num)

    def __eq__(self, other):
        return self.true_value == other.true_value

    def __lt__(self, other):
        return self.true_value < other.true_value

    def __gt__(self, other):
        return self.true_value > other.true_value

    def __ge__(self, other):
        return self.true_value >= other.true_value

    def __le__(self, other):
        return self.true_value <= other.true_value

    def __ne__(self, other):
        return self.true_value != other.true_value



class F1Record(object):
    def __init__(self):
        super().__init__()
        self.value = torch.zeros(4)
    def __iadd__(self, val):
        assert val.size(0) == 1 and val.size(1) == 4
        self.value += val[0]
        return self
    def reset(self,):
        self.value = torch.zeros(4)
    def __str__(self):
        if self.value[0] > 0:
            id_recall = (self.value[2] / self.value[0]).item()
            cls_recall = (self.value[3] / self.value[0]).item()
        else:
            id_recall = cls_recall = 0.
        if self.value[1] > 0:
            id_precision = (self.value[2] / self.value[1]).item()
            cls_precision = (self.value[3] / self.value[1]).item()
        else:
            id_precision = cls_precision = 0.
        if self.value[0] + self.value[1] > 0:
            id_f1 = (self.value[2] * 2 / (self.value[0] + self.value[1])).item()
            cls_f1 = (self.value[3] * 2 / (self.value[0] + self.value[1])).item()
        else:
            id_f1 = cls_f1 = 0.
        display = f"I:{id_f1:.4f};C:{cls_f1:.4f}"
        return display

class Worker(object):

    def __init__(self, opts):
        super().__init__()
        self.train_epoch = opts.train_epoch
        self.no_gpu = opts.no_gpu
        self.gpu = opts.gpu
        self.save_model = opts.save_model
        self.load_model = opts.load_model
        self.log = opts.log
        log_dirs = os.path.split(self.log)[0]
        if not os.path.exists(log_dirs):
            os.makedirs(log_dirs)
        self.log_dir = log_dirs
        logging.basicConfig(filename=self.log, level=logging.INFO)
        self._log = logging.info
        self.epoch = 0
        self.epoch_outputs = dict()
    
    @classmethod
    def from_options(cls, train_epoch:int, no_gpu:bool, gpu:int, save_model:str, load_model:str, log:str):
        class Opts:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        opts = Opts(
            train_epoch = train_epoch,
            no_gpu = no_gpu,
            gpu = gpu,
            save_model = save_model,
            load_model = load_model,
            log = log)
        return cls(opts)

    @classmethod
    def _to_device(cls, instance:Union[torch.Tensor,List[torch.Tensor],Tuple[torch.Tensor,...],Dict[Any,torch.Tensor]], device:Union[torch.device, None]=None):
        if isinstance(instance, list):
            return [cls._to_device(t, device) for t in instance]
        elif isinstance(instance, dict):
            return {key: cls._to_device(value, device=device) for key, value in instance.items()}
        elif isinstance(instance, tuple):
            vals = [cls._to_device(value, device=device) for value in instance]
            return type(instance)(*vals)
        else:
            try:
                return instance.to(device)
            except Exception as e:
                raise(e, f"{type(instance)} not recognized for cuda")

    def run_one_epoch(self, model, loader, split, f_loss=None, optimizer=None, scheduler=None, collect_outputs=None, collect_stats=None, **kwargs):
        if f_loss is None:
            f_loss = model.forward
        if split == "train":
            model.train()
            self.epoch += 1
            if optimizer is None:
                raise ValueError("training requires valid optimizer")
        else:
            model.eval()
        epoch_loss = Record()
        epoch_metric = Record(True)
        epoch_loss.reset()
        epoch_metric.reset()
        if collect_outputs is not None:
            self.epoch_outputs = {key: [] for key in collect_outputs}
        num_batches = len(loader)

        if kwargs is not None:
            info = " ".join([f"{k}: {v}" for k, v in kwargs.items()])
        else:
            info = ""
        iterator = tqdm(loader, f"{self.save_model}|{info}|Epoch {self.epoch:3d}: {split}|", ncols=128)
        for it, batch in enumerate(iterator):
            if not self.no_gpu:
                batch = self._to_device(batch, model.device)
            if split == "train":
                loss = f_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
            else:
                with torch.no_grad():
                    loss = f_loss(batch)
            if loss > 0:
                epoch_loss += loss.item()
                epoch_metric += model.outputs[collect_stats]
            for key in self.epoch_outputs:
                self.epoch_outputs[key].append(model.outputs[key])
            postfix = {"loss": f"{epoch_loss}", "metric": f"{epoch_metric}"}
            iterator.set_postfix(postfix)

        return epoch_loss, epoch_metric

    def save(self,
        model:Union[torch.nn.Module, Dict],
        optimizer:Union[torch.optim.Optimizer, Dict, None]=None,
        scheduler:Union[torch.optim.lr_scheduler._LRScheduler, Dict, None]=None,
        postfix:str=""):

        save_dirs = self.log_dir#os.path.split(self.save_model)[0]
        if not os.path.exists(save_dirs):
            os.makedirs(save_dirs)
        def get_state_dict(x):
            if x is None:
                return None
            elif isinstance(x, dict):
                return x
            else:
                try:
                    return x.state_dict()
                except Exception as e:
                    raise ValueError(f"model, optimizer or scheduler to save must be either a dict or have callable state_dict method")
        if postfix is not "":
            save_model = os.path.join(save_dirs, f"{self.save_model}.{postfix}")
        else:
            save_model = os.path.join(save_dirs, self.save_model)
        torch.save({
            "state_dict": get_state_dict(model),
            "optimizer_state_dict": get_state_dict(optimizer),
            "scheduler_state_dict": get_state_dict(scheduler),
            "iter": self.epoch + 1
            },
            save_model
        )

    def load(self, model:torch.nn.Module, optimizer:Union[torch.optim.Optimizer, None]=None, scheduler:Union[torch.optim.lr_scheduler._LRScheduler,None]=None, path:Union[str, None]=None, load_iter:bool=True, strict:bool=True) -> None:
        if path is None:
            path = self.load_model
        if not os.path.exists(path):
            raise FileNotFoundError(f"the path {path} to saved model is not correct")

        state_dict = torch.load(path, map_location=model.device)
        model.load_state_dict(state_dict=state_dict["state_dict"], strict=strict)
        if load_iter:
            self.epoch = state_dict["iter"] - 1
        if optimizer:
            optimizer.load_state_dict(state_dict=state_dict["optimizer_state_dict"])
        if scheduler:
            scheduler.load_state_dict(state_dict=state_dict["scheduler_state_dict"])
        return None
