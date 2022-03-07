# version description: joint global-local learning with instance-memory loss
# need to maintain and update a jaccard distance matrix during every training epoch
# fusion the clustering results of global and local branches

import collections
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data import build_val_dataloader
from ...models.utils.extract import extract_features
from ...utils.dist_utils import (
    broadcast_tensor,
    broadcast_value,
    get_dist_info,
    synchronize,
)
from .dbscan import label_generator_dbscan, label_generator_dbscan_single  # noqa
from .kmeans import label_generator_kmeans


class LabelGenerator(object):
    """Pseudo Label Generator."""

    __factory = {
        "dbscan": label_generator_dbscan,
        "kmeans": label_generator_kmeans,
    }

    def __init__(
        self, cfg, models, verbose=True  # list of models, e.g. MMT has two models
    ):
        super(LabelGenerator, self).__init__()

        assert (
            "PSEUDO_LABELS" in cfg.TRAIN
        ), "cannot find settings in the config file for pseudo labels"

        self.cfg = cfg
        if isinstance(models, nn.Module):
            models = [models]
        self.models = models
        self.verbose = verbose

        self.data_loaders, self.datasets = build_val_dataloader(
            cfg, for_clustering=True
        )

        self.cluster_type = self.cfg.TRAIN.PSEUDO_LABELS.cluster

        self.num_classes = []
        self.indep_thres = []

        if self.cfg.TRAIN.PSEUDO_LABELS.cluster_num is not None:
            # for kmeans
            self.num_classes = self.cfg.TRAIN.PSEUDO_LABELS.cluster_num

        self.rank, self.world_size, _ = get_dist_info()

        # for regularization term of target loss
        self.dist_cluster = None

    @torch.no_grad()
    def __call__(self, epoch, cuda=True, memory_features=None, **kwargs):

        all_labels = []
        all_centers = []

        for idx, (data_loader, dataset) in enumerate(
            zip(self.data_loaders, self.datasets)
        ):

            # clustering
            try:
                indep_thres = self.indep_thres[idx]
            except Exception:
                indep_thres = None
            try:
                num_classes = self.num_classes[idx]
            except Exception:
                num_classes = None

            if memory_features is None:
                # extract features
                all_features = []
                all_grt_labels = []
                all_camids = []
                for model in self.models:
                    features, grt_labels, camids = extract_features(
                        model,
                        data_loader,
                        dataset,
                        cuda,
                        normalize=self.cfg.TRAIN.PSEUDO_LABELS.norm_feat,
                        with_path=False,
                        for_testing=False,
                        return_labels=True,
                        return_camids=True,
                        prefix="Cluster: ",
                        **kwargs,
                    )
                    all_features.append(features)
                    all_grt_labels.append(grt_labels)
                    all_camids.append(camids)
                all_features = torch.stack(all_features, dim=0).mean(0)
                all_grt_labels = torch.stack(all_grt_labels, dim=0)
                all_camids = torch.stack(all_camids, dim=0)

                if "num_parts" in self.cfg.MODEL:
                    # (Option 1:) original code
                    # num_splits = 0
                    # if self.cfg.MODEL.num_parts > 1:
                    #     num_splits = self.cfg.MODEL.num_parts
                    # if self.cfg.MODEL.include_global:
                    #     num_splits += 1
                    # all_features = torch.split(
                    #     all_features, all_features.size(1) // num_splits, dim=1
                    # )

                    # (Option 2:) for global-local feature extraction, use only global features for cluster
                    # all_features = all_features[0]

                    # (Option 3:) for global-local feature extraction, their mean features for cluster
                    # mean_features = all_features[0]
                    # for i in range(num_splits - 1):
                    #     mean_features += all_features[i+1]
                    # mean_features = mean_features / num_splits
                    # all_features = mean_features

                    # (Option 4:) for global-local feature extraction, use all_features to gain global-centers and local-centers,
                    # cluster once, global branch share pseudo labels
                    num_splits = 0
                    if self.cfg.MODEL.num_parts > 1:
                        num_splits = self.cfg.MODEL.num_parts
                    if self.cfg.MODEL.include_global:
                        num_splits += 1
                    if self.cfg.MODEL.embed_feat == self.cfg.MODEL.loc_embed_feat:
                        # split equally
                        all_features = torch.split(
                            all_features, all_features.size(1) // num_splits, dim=1
                        )
                    elif self.cfg.MODEL.embed_feat == 0 and self.cfg.MODEL.loc_embed_feat > 0:
                        loc_embed_feat = self.cfg.MODEL.loc_embed_feat
                        split_section = []
                        split_section.append(
                            all_features.size(1)-self.cfg.MODEL.num_parts*loc_embed_feat
                        )
                        for i in range(self.cfg.MODEL.num_parts):
                            split_section.append(loc_embed_feat)
                        all_features = torch.split(
                            all_features, split_section, dim=1
                        )
                    else:
                        warnings.warn(
                            "nonsupport combination of MODEL.embed_feat and MODEL.loc_embed_feat"
                        )

            else:
                assert isinstance(memory_features, list)
                all_features = memory_features[idx]

            if self.cfg.TRAIN.PSEUDO_LABELS.norm_feat:
                if isinstance(all_features, list) or isinstance(all_features, tuple) :
                    all_features = [F.normalize(f, p=2, dim=1) for f in all_features]
                else:
                    all_features = F.normalize(all_features, p=2, dim=1)

            if self.rank == 0:
                # for regularization term of target loss, return jaccard distance matrix
                labels, centers, num_classes, indep_thres, dist = self.__factory[
                    self.cluster_type
                ](
                    self.cfg,
                    all_features,
                    all_grt_labels,
                    all_camids,
                    num_classes=num_classes,
                    cuda=cuda,
                    indep_thres=indep_thres,
                )
                self.dist_cluster = dist
                # original code
                # clustering only on GPU:0
                # labels, centers, num_classes, indep_thres = self.__factory[
                #     self.cluster_type
                # ](
                #     self.cfg,
                #     all_features,
                #     num_classes=num_classes,
                #     cuda=cuda,
                #     indep_thres=indep_thres,
                # )

                if self.cfg.TRAIN.PSEUDO_LABELS.norm_center:
                    if isinstance(all_features, list) or isinstance(all_features, tuple) :
                    # global-local feature centers
                        centers = [F.normalize(c, p=2, dim=1) for c in centers]
                    else:
                    # original code
                        centers = F.normalize(centers, p=2, dim=1)

            synchronize()

            # broadcast to other GPUs
            if self.world_size > 1:
                num_classes = int(broadcast_value(num_classes, 0))
                if (
                    self.cfg.TRAIN.PSEUDO_LABELS == "dbscan"
                    and len(self.cfg.TRAIN.PSEUDO_LABELS.eps) > 1
                ):
                    # use clustering reliability criterion
                    indep_thres = broadcast_value(indep_thres, 0)
                if self.rank > 0:
                    labels = torch.arange(len(dataset)).long()
                    if isinstance(all_features, list) or isinstance(all_features, tuple) :
                    # for global-local centers
                        centers = [
                            torch.zeros((num_classes, all_feat.size(-1))).float() for all_feat in all_features
                        ]
                    else:
                    # original code
                        centers = torch.zeros((num_classes, all_features.size(-1))).float()
                labels = broadcast_tensor(labels, 0)
                if isinstance(all_features, list) or isinstance(all_features, tuple) :
                # for global-local feature centers
                    centers = [broadcast_tensor(c, 0) for c in centers]
                else:
                # original code
                    centers = broadcast_tensor(centers, 0)

            try:
                self.indep_thres[idx] = indep_thres
            except Exception:
                self.indep_thres.append(indep_thres)
            try:
                self.num_classes[idx] = num_classes
            except Exception:
                self.num_classes.append(num_classes)

            all_labels.append(labels.tolist())
            all_centers.append(centers)

        self.cfg.TRAIN.PSEUDO_LABELS.cluster_num = self.num_classes

        if self.verbose:
            dataset_names = [
                list(self.cfg.TRAIN.datasets.keys())[i]
                for i in self.cfg.TRAIN.unsup_dataset_indexes
            ]
            for label, dn in zip(all_labels, dataset_names):
                self.print_label_summary(epoch, label, dn)

        return all_labels, all_centers

    def print_label_summary(self, epoch, pseudo_labels, dataset_name):
        # statistics of clusters and un-clustered instances
        index2label = collections.defaultdict(int)
        for label in pseudo_labels:
            index2label[label] += 1
        if -1 in index2label.keys():
            unused_ins_num = index2label.pop(-1)
        else:
            unused_ins_num = 0
        index2label = np.array(list(index2label.values()))
        clu_num = (index2label > 1).sum()
        unclu_ins_num = (index2label == 1).sum()
        print(
            f"\n==> Statistics for {dataset_name} on epoch {epoch}: "
            f"{clu_num} clusters, "
            f"{unclu_ins_num} un-clustered instances, "
            f"{unused_ins_num} unused instances\n"
        )

