# version description: joint Global-Local learning with instance-memory loss (GL)
# need to maintain and update a jaccard distance matrix during every training epoch
# Fusion Clustering results (FC)
# add Soft losses for global and local branches (S)
# with cross-domain Mixup (M)
# add inter-instance soft supervision (I)

import argparse
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path
import warnings

import torch

import collections
import time
import warnings

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from openunreid.apis import BaseRunner, batch_processor, test_reid, set_random_seed
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import build_test_dataloader, build_train_dataloader, build_val_dataloader
from openunreid.models import build_model
from openunreid.models.losses import build_loss
from openunreid.core.metrics.accuracy import accuracy
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.models.utils.extract import extract_features
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger

try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import GradScaler, autocast

    amp_support = True
except:
    amp_support = False

from openunreid.apis.train import batch_processor


class StrongerBaselineRunner(BaseRunner):
    """
    A Stronger Re-ID Base Runner
    """

    def mixup_src_tgt(self, inputs):
        is_mixup = self.cfg.DATA.TRAIN.is_mixup \
                   and (len(self.cfg.TRAIN.datasets) == 2) \
                   and self._epoch >= 1
        mixup_lmd = self.mixup_lmd
        # biased mixup, make sure source data dominate the results
        mixup_lmd = max(mixup_lmd, (1 - mixup_lmd))
        # print("Mixup Lambda: ", mixup_lmd)
        if is_mixup:
            batch_size_per_domain = inputs.size(0) // 2
            #inputs[:batch_size_per_domain] = mixup_lmd * inputs[:batch_size_per_domain] + \
            #                                 (1 - mixup_lmd) * inputs[batch_size_per_domain:]
            #inputs[batch_size_per_domain:] = mixup_lmd * inputs[batch_size_per_domain:] + \
            #                                 (1 - mixup_lmd) * inputs[:batch_size_per_domain]
            """
            temp = inputs
            inputs[:batch_size_per_domain] = mixup_lmd * temp[:batch_size_per_domain] + \
                                             (1 - mixup_lmd) * temp[batch_size_per_domain:]
            inputs[batch_size_per_domain:] = mixup_lmd * temp[batch_size_per_domain:] + \
                                             (1 - mixup_lmd) * temp[:batch_size_per_domain]
            """

            # SourceMix + Source + Target + TargetMix
            batch_size_mix = batch_size_per_domain // 2
            temp = inputs
            inputs[:batch_size_mix] = \
                mixup_lmd * temp[:batch_size_mix] + \
                (1 - mixup_lmd) * temp[batch_size_mix*2:batch_size_mix*3]
            inputs[batch_size_mix*3:] = \
                mixup_lmd * temp[batch_size_mix*3:] + \
                (1 - mixup_lmd) * temp[batch_size_mix:batch_size_mix*2]
        return inputs, is_mixup, mixup_lmd

    def train_step(self, iter, batch):
        # need to be re-written case by case
        assert not isinstance(
            self.model, list
        ), "please re-write 'train_step()' to support list of models"

        data = batch_processor(batch, self.cfg.MODEL.dsbn)
        if len(data["img"]) > 1:
            warnings.warn(
                "please re-write the 'runner.train_step()' function to make use of "
                "mutual transformer."
            )

        inputs = data["img"][0].cuda()
        targets = data["id"].cuda()
        indexes = data["ind"].cuda()

        # (Optional:) cross-domain mixup
        inputs, is_mixup, mixup_lmd = self.mixup_src_tgt(inputs)
        # print("Mixup lambda: ", mixup_lmd)

        results = self.model(inputs)

        # for global-local branches
        num_splits = 0
        if "num_parts" in self.cfg.MODEL:
            if self.cfg.MODEL.num_parts > 1:
                num_splits = self.cfg.MODEL.num_parts
                if self.cfg.MODEL.include_global is True:
                    num_splits += 1

        if "prob" in results.keys():
            # for global-local feature extraction
            if num_splits > 0:
                for i in range(num_splits):
                    (results["prob"])[i] = (results["prob"])[i][
                                      :, : self.train_loader.loader.dataset.num_pids
                                      ]
            else:
                results["prob"] = results["prob"][
                                              :, : self.train_loader.loader.dataset.num_pids
                                              ]

        total_loss = 0
        meters = {}

        # lmd = 1.0
        lmd = self.cfg.TRAIN.LOSS.lmd_local
        batch_size_per_domain = int(targets.size(0)/2)

        # separate losses of source and target data from joint flow,
        # can trade-off their respective importance,
        # may influence the triplet mining
        indep_losses = False and (len(self.cfg.TRAIN.datasets) == 2)
        lmd_src = 0.5
        lmd_tgt = 1.0 - lmd_src

        # domain-independent prob for classification loss and triplet loss
        indep_prob = False and (len(self.cfg.TRAIN.datasets) == 2)
        # msmt17/market as source domain: 1041/751 classes
        src_num_id = 1041

        # for global-local branches
        if num_splits > 0:
            branches_results = []
            for i in range(num_splits):
                branch_results = {}
                for result_key in results.keys():
                    branch_results[result_key] = (results[result_key])[i]
                branches_results.append(branch_results)

            # global branch losses:
            if self.cfg.MODEL.include_global:
                lmd_sce_glo = 0.0
                lmd_stri_glo = 0.0
                lmd_iisce_glo = 0.5

                for key in self.criterions.keys():
                    # (Optional:) regularization term of target loss, concat all global and local embeddings
                    if key == "instance_memory":
                        loss = self.criterions[key](
                            results, indexes,
                            self.label_generator.dist_cluster,
                            self._epoch
                        )
                    # (Optional: ) soft classification losses with local branch classification prediction as targets
                    elif key == "soft_entropy":
                        '''
                        assert num_splits > 2
                        loss = 0
                        loss += lmd_sce_glo * self.criterions[key](
                            branches_results[0], branches_results[1]
                        ) / (num_splits - 1)
                        for m in range(num_splits - 2):
                            loss += lmd_sce_glo * self.criterions[key](
                                branches_results[0], branches_results[m + 2]
                            ) / (num_splits - 1)
                        '''
                        assert num_splits == 3
                        loss = lmd_sce_glo * self.criterions[key](
                            branches_results[0], branches_results[1], branches_results[2]
                        )
                    # (Optional:) soft triplet loss with local branches embeddings as supervision
                    elif key == "soft_softmax_triplet":
                        assert num_splits > 2
                        loss = 0
                        loss += lmd_stri_glo * self.criterions[key](
                            branches_results[0], targets, branches_results[1]
                        ) / (num_splits - 1)
                        for n in range(num_splits - 2):
                            loss += lmd_stri_glo * self.criterions[key](
                                branches_results[0], targets, branches_results[n + 2]
                            ) / (num_splits - 1)
                    else:
                        if indep_prob and "prob" in branches_results[0].keys():
                            branches_results[0]["prob"][:batch_size_per_domain, src_num_id:].data.copy_(
                                torch.zeros(
                                    batch_size_per_domain, branches_results[0]["prob"].size(-1) - src_num_id
                                ).float().to(branches_results[0]["prob"].device)
                            )
                            branches_results[0]["prob"][batch_size_per_domain:, :src_num_id].data.copy_(
                                torch.zeros(
                                    batch_size_per_domain, src_num_id
                                ).float().to(branches_results[0]["prob"].device)
                            )
                        if indep_losses:
                            loss = lmd_src * self.criterions[key]({
                                    key:value[:batch_size_per_domain,:] for key, value in branches_results[0].items()
                                }, targets[:batch_size_per_domain]) + lmd_tgt * self.criterions[key]({
                                    key:value[batch_size_per_domain:,:] for key, value in branches_results[0].items()
                                }, targets[batch_size_per_domain:])
                        else:
                            if is_mixup:
                                loss = self.criterions[key](
                                    branches_results[0], targets, is_mixup=is_mixup, mixup_lmd=mixup_lmd
                                )
                            else:
                                if key == "cross_entropy":
                                    '''
                                    loss = (1 - lmd_sce_glo) * (1 - lmd_iisce_glo) \
                                           * self.criterions[key](branches_results[0], targets)
                                    '''
                                    loss = (1 - lmd_sce_glo) * self.criterions[key](branches_results[0], targets)
                                elif key == "inter_instance_soft_entropy":
                                    loss = lmd_iisce_glo * self.criterions[key](branches_results[0], targets)
                                else:
                                    loss = (1 - lmd_stri_glo) * self.criterions[key](branches_results[0], targets)
                    total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
                    meters[key] = loss.item()

            # local branches losses:
            if num_splits > 1:
                lmd_sce_loc = 1.0
                lmd_iisce_loc = 0.0
                for j in range(num_splits-1):
                    for key in self.criterions.keys():
                        if key == "instance_memory":
                            continue
                        # (Optional: ) soft classification losses with global branch classification prediction as targets
                        elif key == "soft_entropy":
                            loss = lmd_sce_loc * self.criterions[key](
                                branches_results[j + 1], branches_results[0]
                            ) * lmd
                        # (Optional:) soft triplet loss with global branches embeddings as supervision
                        elif key == "soft_softmax_triplet":
                            loss = 0.0 * self.criterions[key](
                                branches_results[j + 1], targets, branches_results[0]
                            )
                        # (Optional:) whether employ triplet loss in local branches
                        elif key == "softmax_triplet" and not self.cfg.TRAIN.LOSS.include_local_triplet_loss:
                            continue
                        else:
                            if indep_prob and "prob" in branches_results[j + 1].keys():
                                branches_results[j + 1]["prob"][:batch_size_per_domain, src_num_id:].data.copy_(
                                    torch.zeros(
                                        batch_size_per_domain, branches_results[0]["prob"].size(-1) - src_num_id
                                    ).float().to(branches_results[0]["prob"].device)
                                )
                                branches_results[j + 1]["prob"][batch_size_per_domain:, :src_num_id].data.copy_(
                                    torch.zeros(
                                        batch_size_per_domain, src_num_id
                                    ).float().to(branches_results[0]["prob"].device)
                                )
                            if indep_losses:
                                loss = lmd_src * self.criterions[key]({
                                    key: value[:batch_size_per_domain, :] for key, value in branches_results[j + 1].items()
                                }, targets[:batch_size_per_domain]) + lmd_tgt * self.criterions[key]({
                                    key: value[batch_size_per_domain:, :] for key, value in branches_results[j + 1].items()
                                }, targets[batch_size_per_domain:])
                            else:
                                if is_mixup:
                                    loss = self.criterions[key](
                                        branches_results[j + 1], targets, is_mixup=is_mixup, mixup_lmd=mixup_lmd
                                    ) * lmd
                                else:
                                    if key == "inter_instance_soft_entropy":
                                        loss = lmd_iisce_loc * self.criterions[key](branches_results[j + 1], targets) * lmd
                                    else:
                                        loss = (1 - lmd_sce_loc) * self.criterions[key](branches_results[j + 1], targets) * lmd
                        total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
                        # meters[key+"_loc_"+"%d" % (j+1)] = loss.item()
                        meters[key] += loss.item()

            if "prob" in branches_results[0].keys():
                acc = accuracy(branches_results[0]["prob"].data, targets.data)
                meters["Acc@1"] = acc[0]

            self.train_progress.update(meters)

        else:
            # original code
            for key in self.criterions.keys():
                # for regularization term of target loss
                if key == 'instance_memory':
                    loss = self.criterions[key](results, indexes, self.label_generator.dist_cluster, self._epoch)
                # original code
                else:
                    loss = self.criterions[key](results, targets)
                total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
                meters[key] = loss.item()

            if "prob" in results.keys():
                acc = accuracy(results["prob"].data, targets.data)
                meters["Acc@1"] = acc[0]

            self.train_progress.update(meters)

        return total_loss


def parge_config():
    parser = argparse.ArgumentParser(description="A stronger cluster baseline training")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--work-dir", help="the dir to save logs and models", default=""
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    if not args.work_dir:
        args.work_dir = Path(args.config).stem
    cfg.work_dir = cfg.LOGS_ROOT / args.work_dir
    mkdir_if_missing(cfg.work_dir)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    shutil.copy(args.config, cfg.work_dir / "config.yaml")

    return args, cfg


def main():
    start_time = time.monotonic()

    # init distributed training
    args, cfg = parge_config()
    dist = init_dist(cfg)
    set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)
    synchronize()

    # init logging file
    logger = Logger(cfg.work_dir / "log.txt", debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build train loader
    train_loader, train_sets = build_train_dataloader(cfg)

    # the number of classes for the model is tricky,
    # you need to make sure that
    # it is always larger than the number of clusters
    num_classes = 0
    # for regularization term in target loss
    num_memory = 0
    for idx, set in enumerate(train_sets):
        if idx in cfg.TRAIN.unsup_dataset_indexes:
            # number of clusters in an unsupervised dataset
            # must not be larger than the number of images
            num_classes += len(set)
            # for regularization term in target loss,
            # only unsupervised datasets are counted
            num_memory += len(set)
        else:
            # ground-truth classes for supervised dataset
            num_classes += set.num_pids

    # build model
    model = build_model(cfg, num_classes, init=cfg.MODEL.source_pretrained)
    model.cuda()

    if dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.gpu],
            output_device=cfg.gpu,
            find_unused_parameters=True,
        )
    elif cfg.total_gpus > 1:
        model = torch.nn.DataParallel(model)

    # build optimizer
    optimizer = build_optimizer([model,], **cfg.TRAIN.OPTIM)

    # build lr_scheduler
    if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
        lr_scheduler = build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)
    else:
        lr_scheduler = None

    # for regularization term in target loss
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        num_features = model.module.num_features
    else:
        num_features = model.num_features
    if cfg.MODEL.embed_feat > 0:
        num_features = cfg.MODEL.embed_feat
    if cfg.MODEL.num_parts > 0 and cfg.MODEL.include_global:
        if cfg.MODEL.loc_embed_feat > 0:
            num_features += (cfg.MODEL.num_parts * cfg.MODEL.loc_embed_feat)
        else:
            num_features += (cfg.MODEL.num_parts * num_features)


    # build loss functions
    # for regularization term in target loss
    criterions = build_loss(cfg.TRAIN.LOSS,
                            num_classes=num_classes,
                            num_features=num_features,
                            num_memory=num_memory,
                            cuda=True)
    # original code
    # criterions = build_loss(cfg.TRAIN.LOSS, num_classes=num_classes, cuda=True)

    # init instance memory
    if 'instance_memory' in criterions.keys():
        loaders, datasets = build_val_dataloader(
            cfg, for_clustering=True, all_datasets=True
        )
        memory_features = []
        for idx, (loader, dataset) in enumerate(zip(loaders, datasets)):
            features = extract_features(
                model, loader, dataset, with_path=False, prefix="Extract: ",
            )
            assert features.size(0) == len(dataset)
            if idx in cfg.TRAIN.unsup_dataset_indexes:
                # init memory for unlabeled data with instance features
                memory_features.append(features)
            # else:
            #     # init memory for labeled data with class centers
            #     centers_dict = collections.defaultdict(list)
            #     for i, (_, pid, _) in enumerate(dataset):
            #         centers_dict[pid].append(features[i].unsqueeze(0))
            #     centers = [
            #         torch.cat(centers_dict[pid], 0).mean(0)
            #         for pid in sorted(centers_dict.keys())
            #     ]
            #     memory_features.append(torch.stack(centers, 0))
        del loaders, datasets

        memory_features = torch.cat(memory_features)
        criterions["instance_memory"]._update_feature(memory_features)

    # build runner
    runner = StrongerBaselineRunner(
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        train_sets=train_sets,
        lr_scheduler=lr_scheduler,
        reset_optim=True,
    )

    # resume
    if args.resume_from:
        runner.resume(args.resume_from)

    # start training
    runner.run()

    # load the best validation model
    runner.resume(cfg.work_dir / "model_best.pth")

    # final testing with best validation model
    test_loaders, queries, galleries = build_test_dataloader(cfg)
    for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):
        cmc, mAP = test_reid(
            cfg, model, loader, query, gallery, dataset_name=cfg.TEST.datasets[i]
        )

    # load the final model
    runner.resume(cfg.work_dir / "checkpoint.pth")

    # final testing with final model
    test_loaders, queries, galleries = build_test_dataloader(cfg)
    for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):
        cmc, mAP = test_reid(
            cfg, model, loader, query, gallery, dataset_name=cfg.TEST.datasets[i]
        )



    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


if __name__ == "__main__":
    main()
