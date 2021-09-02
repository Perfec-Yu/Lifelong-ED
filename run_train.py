import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
from tqdm import tqdm

from utils.optimizer import AdamW
from utils.options import parse_arguments
from utils.datastream import get_stage_loaders, get_stage_loaders_n
from utils.worker import Worker
from models.nets import LInEx, BIC, ICARL

PERM = [[0, 1, 2, 3,4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def by_class(preds, labels, learned_labels=None):
    match = (preds == labels).float()
    nlabels = max(torch.max(labels).item(), torch.max(preds).item())
    bc = {}

    ag = 0; ad = 0; am = 0
    for label in range(1, nlabels+1):
        lg = (labels==label); ld = (preds==label)
        lr = torch.sum(match[lg]) / torch.sum(lg.float())
        lp = torch.sum(match[ld]) / torch.sum(ld.float())
        lf = 2 * lr * lp / (lr + lp)
        if torch.isnan(lf):
            bc[label] = (0, 0, 0)
        else:
            bc[label] = (lp.item(), lr.item(), lf.item())
        if learned_labels is not None and label in learned_labels:
            ag += lg.float().sum()
            ad += ld.float().sum()
            am += match[lg].sum()
    if learned_labels is None:
        ag = (labels!=0); ad = (preds!=0)
        sum_ad = torch.sum(ag.float())
        if sum_ad == 0:
            ap = ar = 0
        else:
            ar = torch.sum(match[ag]) / torch.sum(ag.float())
            ap = torch.sum(match[ad]) / torch.sum(ad.float())
    else:
        if ad == 0:
            ap = ar = 0
        else:
            ar = am / ag; ap = am / ad
    if ap == 0:
        af = ap = ar = 0
    else:
        af = 2 * ar * ap / (ar + ap)
        af = af.item(); ar = ar.item(); ap = ap.item()
    return bc, (ap, ar, af)


def main():
    
    opts = parse_arguments()
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    summary = SummaryWriter(opts.log_dir)

    dataset_id = 0
    if 'MAVEN' in opts.log_dir:
        dataset_id = 0
    elif 'ACE' in opts.log_dir:
        dataset_id = 1

    perm_id = opts.perm_id
    if opts.setting == "classic":
        streams = json.load(open(opts.stream_file))
        streams = [streams[t] for t in PERM[perm_id]]
        loaders, exemplar_loaders, stage_labels, label2id = get_stage_loaders(root=opts.json_root,
            feature_root=opts.feature_root,
            batch_size=opts.batch_size,
            streams=streams,
            num_workers=8,
            dataset=dataset_id)
    else:
        sis = json.load(open("data/MAVEN/stream_instances_2227341903.json"))
        if perm_id <= 3:
            print(f"running perm {perm_id}")
            sis = [sis[t] for t in PERM[perm_id]]
        loaders, exemplar_loaders, stage_labels, label2id = get_stage_loaders_n(root=opts.json_root,
            feature_root=opts.feature_root,
            batch_size=opts.batch_size,
            streams=json.load(open(opts.stream_file)),
            streams_instances=sis,
            num_workers=8,
            dataset=dataset_id)
    if opts.balance == 'bic':
        model = BIC(
            nhead=opts.nhead,
            nlayers=opts.nlayers,
            input_dim=opts.input_dim,
            hidden_dim=opts.hidden_dim,
            max_slots=opts.max_slots,
            init_slots=max(stage_labels[0])+1 if not opts.test_only else max(stage_labels[-1])+1,
            device=torch.device(torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))
        )
    elif opts.balance == "icarl":
        model = ICARL(
            nhead=opts.nhead,
            nlayers=opts.nlayers,
            input_dim=opts.input_dim,
            hidden_dim=opts.hidden_dim,
            max_slots=opts.max_slots,
            init_slots=max(stage_labels[0])+1 if not opts.test_only else max(stage_labels[-1])+1,
            device=torch.device(torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))
        )
    else:
        model = LInEx(
            nhead=opts.nhead,
            nlayers=opts.nlayers,
            input_dim=opts.input_dim,
            hidden_dim=opts.hidden_dim,
            max_slots=opts.max_slots,
            init_slots=max(stage_labels[0])+1 if not opts.test_only else max(stage_labels[-1])+1,
            device=torch.device(torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))
        )
    param_groups = [
        {"params": [param for name, param in model.named_parameters() if param.requires_grad and 'correction' not in name],
        "lr":opts.learning_rate,
        "weight_decay": opts.decay,
        "betas": (0.9, 0.999)}
        ]
    optimizer = AdamW(params=param_groups)
    optimizer_correction = None
    if opts.balance == "bic":
        correction_param_groups = [
            {"params": [param for name, param in model.named_parameters() if param.requires_grad and "correction_weight" in name],
            "lr":opts.learning_rate,
            "weight_decay": 0,
            "betas": (0.9, 0.999)},
            {"params": [param for name, param in model.named_parameters() if param.requires_grad and "correction_bias" in name],
            "lr":opts.learning_rate,
            "weight_decay": 0.01,
            "betas": (0.9, 0.999)}
        ]
        assert len(correction_param_groups[0]['params']) == 1
        assert len(correction_param_groups[1]['params']) == 1
        optimizer_correction = AdamW(params=correction_param_groups)
    worker = Worker(opts)
    worker._log(str(opts))
    worker._log(str(label2id))
    if opts.test_only:
        worker.load(model)
    best_dev = best_test = None
    collect_stats = "accuracy"
    collect_outputs = {"prediction", "label"}
    termination = False
    patience = opts.patience
    no_better = 0
    loader_id = 0
    total_epoch = 0
    none_mul = 4
    learned_labels = set(stage_labels[0])
    dev_metrics = None
    test_metrics = None
    while not termination:
        if not opts.test_only:
            if opts.skip_first and loader_id == 0:
                worker.load(model, optimizer, path=opts.load_first, strict=opts.balance!='bic')
                total_epoch += worker.epoch
            elif opts.skip_second and loader_id == 1:
                worker.load(model, optimizer, path=opts.load_second, strict=opts.balance!='bic')
                total_epoch += worker.epoch
            else:
                if opts.finetune:
                    train_loss = lambda batch:model.forward(batch)
                elif opts.balance == "bic" and loader_id >= 2:
                    train_loss = lambda batch:model.forward(batch, bias_correction="last", exemplar=True, exemplar_distill=True, distill=True, tau=0.5)
                elif opts.balance == "fd":
                    train_loss = lambda batch:model.forward(batch, exemplar=True, feature_distill=True, exemplar_distill=True, distill=True, tau=0.5)
                elif opts.balance == "mul":
                    train_loss = lambda batch:model.forward(batch, exemplar=True, mul_distill=True, exemplar_distill=True, distill=True, tau=0.5)
                else:
                    train_loss = lambda batch:model.forward(batch, exemplar=True, exemplar_distill=True, distill=True, tau=0.5)
                epoch_loss, epoch_metric = worker.run_one_epoch(
                    model=model,
                    f_loss=train_loss,
                    loader=loaders[loader_id],
                    split="train",
                    optimizer=optimizer,
                    collect_stats=collect_stats,
                    prog=loader_id)
                total_epoch += 1

                for output_log in [print, worker._log]:
                    output_log(
                        f"Epoch {worker.epoch:3d}  Train Loss {epoch_loss} {epoch_metric}")
        else:
            learned_labels = set([t for stream in stage_labels for t in stream])
            termination = True

        if opts.test_only:
            if opts.balance == "icarl":
                exemplar = model.set_exemplar(exemplar_loaders[loader_id], output_only=True)
                exemplar_features = []
                exemplar_labels = []
                for label, features in exemplar.items():
                    exemplar_features.append(features)
                    exemplar_labels.extend([label]*features.size(0))
                exemplar_features = torch.cat(exemplar_features, dim=0).cpu()
                exemplar_labels = torch.LongTensor(exemplar_labels).cpu()
                if model.exemplar_features is not None:
                    exemplar_features = torch.cat((model.exemplar_features, exemplar_features), dim=0)
                    exemplar_labels = torch.cat((model.exemplar_labels, exemplar_labels), dim=0)
                model.set_none_feat(loaders[loader_id])
                score_fn = lambda t:model.score(t, exemplar=(exemplar_labels, exemplar_features))
            else:
                score_fn = model.score

            dev_loss, dev_metrics = worker.run_one_epoch(
                model=model,
                f_loss=score_fn,
                loader=loaders[-2],
                split="dev",
                collect_stats=collect_stats,
                collect_outputs=collect_outputs)
            dev_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
            dev_scores, (dev_p, dev_r, dev_f) = by_class(dev_outputs["prediction"], dev_outputs["label"], learned_labels=learned_labels)
            dev_class_f1 = {k: dev_scores[k][2] for k in dev_scores}
            for k,v in dev_class_f1.items():
                add_summary_value(summary, f"dev_class_{k}", v, total_epoch)
            dev_metrics = dev_f
            for output_log in [print, worker._log]:
                output_log(
                    f"Epoch {worker.epoch:3d}:  Dev {dev_metrics}"
                )
            test_loss, test_metrics = worker.run_one_epoch(
                model=model,
                f_loss=score_fn,
                loader=loaders[-1],
                split="test",
                collect_stats=collect_stats,
                collect_outputs=collect_outputs)
            test_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
            torch.save(test_outputs, f"log/{os.path.basename(opts.load_model)}.output")
            test_scores, (test_p, test_r, test_f) = by_class(test_outputs["prediction"], test_outputs["label"], learned_labels=learned_labels)
            test_class_f1 = {k: test_scores[k][2] for k in test_scores}
            for k,v in test_class_f1.items():
                add_summary_value(summary, f"test_class_{k}", v, total_epoch)
            test_metrics = test_f
            for output_log in [print, worker._log]:
                output_log(
                    f"Epoch {worker.epoch:3d}: Test {test_metrics}"
                )
            if opts.test_only:
                frequency = {}
                for loader in loaders[:-2]:
                    indices = loader.dataset.label2index
                    for label in indices.keys():
                        if label != 0:
                            frequency[label] = indices[label][1] - indices[label][0]
                with open("data/MAVEN/label2id.json") as fp:
                    name2label = json.load(fp)
                    label2name = {v:k for k,v in name2label.items()}
                id2label = {v:k for k,v in label2id.items()}
                sf = [(frequency[l], label2name[id2label[l] ], dev_class_f1[l], test_class_f1[l]) for l in frequency]
                sf.sort(key=lambda t:t[0])
                print("macro:", sum([t[3] for t in sf]) / len(sf))

        if not opts.test_only:
            if best_dev is None or dev_metrics > best_dev:
                best_dev = dev_metrics
                worker.save(model, optimizer, postfix=str(loader_id))
                best_test = test_metrics
                no_better = 0
            else:
                no_better += 1
            print(f"patience: {no_better} / {patience}")

            if (no_better == patience) or (worker.epoch == worker.train_epoch) or (opts.skip_first and loader_id == 0) or (opts.skip_second and loader_id == 1):
                loader_id += 1
                no_better = 0
                worker.load(model, optimizer, path=os.path.join(opts.log_dir, f"{worker.save_model}.{loader_id-1}"))
                if not opts.finetune:
                    print("setting train exemplar for learned classes")
                    model.set_exemplar(exemplar_loaders[loader_id-1])
                if opts.balance == "icarl":
                    model.set_none_feat(loaders[loader_id-1])
                elif opts.balance == "bic" and loader_id >= 2:
                    # train stream, release next stream
                    # only apply to finish second round training (loader_id >= 1 + 1 = 2)
                    print("setting dev exemplar for learned classes")
                    model.set_exemplar(loaders[-2], q=5, label_sets=stage_labels[loader_id-1], output="dev")
                    cur_dev_exe_f, cur_dev_exe_l= model.dev_exemplar_features, model.dev_exemplar_labels
                    print("sample none instances for bic training")
                    none_exemplar = model.set_exemplar(loaders[-2], q=int(none_mul*cur_dev_exe_f.size(0)), label_sets=[0], collect_none=True, output_only=True, output=None)
                    cur_exe_f = torch.cat((none_exemplar[0], cur_dev_exe_f), dim=0)
                    cur_exe_l = torch.cat((torch.zeros(none_exemplar[0].size(0)).to(cur_dev_exe_l), cur_dev_exe_l), dim=0)
                    dev_bic_loader = DataLoader(
                        TensorDataset(cur_exe_f, cur_exe_l),
                        batch_size=128,
                        shuffle=True,
                        drop_last=False,
                        num_workers=8
                        )
                    for _bias_epoch in range(2):
                        with torch.autograd.set_detect_anomaly(True):
                            worker.run_one_epoch(
                                model=model,
                                f_loss=model.forward_correction,
                                loader=dev_bic_loader,
                                split="train",
                                optimizer=optimizer_correction,
                                collect_stats=collect_stats,
                                prog=loader_id,
                                run='bic dev')
                elif opts.balance == "eeil" and loader_id >= 2:
                    cur_exe_f, cur_exe_l= model.exemplar_features, model.exemplar_labels
                    none_exemplar = model.set_exemplar(loaders[loader_id-1], q=int(none_mul*cur_exe_f.size(0)), label_sets=[0], collect_none=True, output_only=True, output=None)
                    cur_exe_f = torch.cat((none_exemplar[0], cur_exe_f), dim=0)
                    cur_exe_l = torch.cat((torch.zeros(none_exemplar[0].size(0)).to(cur_exe_l), cur_exe_l), dim=0)
                    eeil_train_loader = DataLoader(
                        TensorDataset(cur_exe_f, cur_exe_l),
                        batch_size=128,
                        shuffle=True,
                        drop_last=False,
                        num_workers=8
                        )
                    for i in range(5):
                        worker.run_one_epoch(
                            model=model,
                            f_loss=model.forward,
                            loader=eeil_train_loader,
                            split="train",
                            optimizer=optimizer,
                            collect_stats=collect_stats,
                            prog=loader_id,
                            run='eeil balance')
                if opts.balance in ['eeil', 'bic']:
                    worker.save(model, optimizer, postfix=str(loader_id-1))
                    dev_loss, dev_metrics = worker.run_one_epoch(
                        model=model,
                        f_loss=model.score,
                        loader=loaders[-2],
                        split="dev",
                        collect_stats=collect_stats,
                        collect_outputs=collect_outputs)
                    dev_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
                    dev_scores, (dev_p, dev_r, dev_f) = by_class(dev_outputs["prediction"], dev_outputs["label"], learned_labels=learned_labels)
                    dev_class_f1 = {k: dev_scores[k][2] for k in dev_scores}
                    for k,v in dev_class_f1.items():
                        add_summary_value(summary, f"dev_class_{k}", v, total_epoch)
                    dev_metrics = dev_f
                    test_loss, test_metrics = worker.run_one_epoch(
                        model=model,
                        loader=loaders[-1],
                        f_loss=model.score,
                        split="test",
                        collect_stats=collect_stats,
                        collect_outputs=collect_outputs)
                    test_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
                    test_scores, (test_p, test_r, test_f) = by_class(test_outputs["prediction"], test_outputs["label"], learned_labels=learned_labels)
                    test_class_f1 = {k: test_scores[k][2] for k in test_scores}
                    for k,v in test_class_f1.items():
                        add_summary_value(summary, f"test_class_{k}", v, total_epoch)
                    test_metrics = test_f
                    best_dev = dev_metrics; best_test = test_metrics
                if not opts.finetune:
                    model.set_history()
                for output_log in [print, worker._log]:
                    output_log(f"BEST DEV {loader_id-1}: {best_dev if best_dev is not None else 0}")
                    output_log(f"BEST TEST {loader_id-1}: {best_test if best_test is not None else 0}")
                if loader_id == len(loaders) - 2:
                    termination = True
                else:
                    learned_labels = learned_labels.union(set(stage_labels[loader_id]))
                    if opts.balance == 'bic':
                        model.correction_stream.append(max(learned_labels) + 1)
                    if opts.kt:
                        next_exemplar = model.set_exemplar(exemplar_loaders[loader_id], output_only=True)
                        next_frequency = {}
                        indices = loaders[loader_id].dataset.label2index
                        for label in stage_labels[loader_id]:
                            if label != 0:
                                next_frequency[label] = indices[label][1] - indices[label][0]
                        if opts.kt2:
                            next_inits = model.initialize2(
                                exemplar=next_exemplar,
                                ninstances=next_frequency,
                                gamma=opts.kt_gamma,
                                tau=opts.kt_tau,
                                alpha=opts.kt_alpha,
                                delta=opts.kt_delta)
                        else:
                            next_inits = model.initialize(
                                exemplar=next_exemplar,
                                ninstances=next_frequency,
                                gamma=opts.kt_gamma,
                                tau=opts.kt_tau,
                                alpha=opts.kt_alpha)
                        torch.save(model.outputs["new2old"], os.path.join(opts.log_dir, f"{loader_id}_to_{loader_id-1}"))
                        model.extend(next_inits)
                        assert model.nslots == max(learned_labels) + 1
                    else:
                        model.nslots = max(learned_labels) + 1
                worker.epoch = 0
                best_dev = None; best_test = None

if __name__ == "__main__":
    main()
