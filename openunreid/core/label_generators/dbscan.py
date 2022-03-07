# version description: joint global-local learning with instance-memory loss
# need to maintain and update a jaccard distance matrix during every training epoch
# fusion the clustering results of global and local branches

import collections
import warnings
import time

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from ...utils.torch_utils import to_torch, to_numpy
from ..utils.compute_dist import build_dist

__all__ = ["label_generator_dbscan_single", "label_generator_dbscan"]


@torch.no_grad()
def label_generator_dbscan_single(cfg, features, dist, eps, **kwargs):
    assert isinstance(dist, np.ndarray)

    # clustering
    min_samples = cfg.TRAIN.PSEUDO_LABELS.min_samples
    use_outliers = cfg.TRAIN.PSEUDO_LABELS.use_outliers

    cluster = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1,)
    labels = cluster.fit_predict(dist)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # cluster labels -> pseudo labels
    # compute cluster centers
    if isinstance(features, list):
        # global-local feature centers
        centers = [collections.defaultdict(list) for _ in range(len(features))]
    else:
        # original code
        centers = collections.defaultdict(list)
    outliers = 0
    for i, label in enumerate(labels):
        if label == -1:
            if not use_outliers:
                continue
            labels[i] = num_clusters + outliers
            outliers += 1

        if isinstance(features, list):
        # update global-local feature centers
            for j in range(len(features)):
                centers[j][labels[i]].append(features[j][i])
        else:
        # original code
            centers[labels[i]].append(features[i])

    if isinstance(features, list):
    # for global-local feature centers
        for i in range(len(features)):
            centers[i] = [
                torch.stack(centers[i][idx], dim=0).mean(0) for idx in sorted(centers[i].keys())
            ]
            centers[i] = torch.stack(centers[i], dim=0)
    else:
    # original code
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]
        centers = torch.stack(centers, dim=0)
    labels = to_torch(labels).long()
    num_clusters += outliers

    return labels, centers, num_clusters


@torch.no_grad()
def  label_generator_dbscan(cfg, features, ground_truth_labels, camids=None, cuda=True, indep_thres=None, **kwargs):
    assert cfg.TRAIN.PSEUDO_LABELS.cluster == "dbscan"

    if not cuda:
        cfg.TRAIN.PSEUDO_LABELS.search_type = 3

    # independent clustering for every global and local branches
    if cfg.TRAIN.PSEUDO_LABELS.cluster_per_branch and \
            cfg.MODEL.num_parts > 0 and \
            cfg.MODEL.include_global:
        # compute distance matrix by features
        assert (
                isinstance(features, list)
        ), "independent clustering require both global and local features"
        # dist = [
        #     build_dist(cfg.TRAIN.PSEUDO_LABELS, feat, verbose=True) for feat in features
        # ]
        dist = build_dist(cfg.TRAIN.PSEUDO_LABELS, features[0], verbose=True)
        dist = [dist for i in range(len(features))]

        if camids is not None:
            N = camids.numel()
            camids_sim = (
                camids.expand(N, N).eq(camids.expand(N, N).t()).float()
            )
            # Compute intra-camera compensation distance
            option = 3
            if option == 1:
                """(Option 1)
                Apply static intra-camera compensation distance
                based on a predefined hyper-parameter
                """
                lambda_iccd = 0.03
                intra_cam_comp_dist = (lambda_iccd * camids_sim).cpu().numpy()
                dist = [d + intra_cam_comp_dist for d in dist]
            elif option == 2:
                """(Option 2) 
                Apply intra-camera compensation distance
                based on global and local feature distance 
                and a predefined compensation hyper-parameter
                """
                lambda_local= 0.2
                dist[0] = (1 - lambda_local) * dist[0] + \
                          lambda_local * (dist [1] + dist[2]) / (len(features) - 1)
                lambda_iccd = 0.03
                dist[0] =  dist[0] + (lambda_iccd * camids_sim).cpu().numpy()
            elif option == 3 or option == 4 or option == 5:
                """(Option 3/4/5) 
                Apply: dynamic intra-camera compensation / dynamic cross-camera encouragement / both distance
                based on difference between intra-camera k-nearest neighbors
                and cross-camera k-nearest neighbors
                """
                ki = 3    # number of intra-camera knn
                kc = 3    # number of cross-camera knn

                dist[0] = to_torch(dist[0])
                mask = torch.ones(dist[0].size()).float()

                dist_same_camid = torch.where(camids_sim == 1, dist[0], mask)
                _, index_sorted_same_camid = torch.sort(dist_same_camid, dim=1, descending=False)
                index_selected_same_camid = torch.zeros(dist[0].size())
                index_selected_same_camid.scatter_(
                    1, index_sorted_same_camid[:, 0:ki], torch.ones(dist[0].size(0), ki).float()
                )
                # Average distance of intra-camera knn
                dist_icknn = torch.sum(index_selected_same_camid * dist[0], dim=1) / ki

                dist_diff_camid = torch.where(camids_sim == 0, dist[0], mask)
                _, index_sorted_diff_camid = torch.sort(dist_diff_camid, dim=1, descending=False)
                index_selected_diff_camid = torch.zeros(dist[0].size())
                index_selected_diff_camid.scatter_(
                    1, index_sorted_diff_camid[:, 0:kc], torch.ones(dist[0].size(0), kc).float()
                )
                # Average distance of cross-camera knn
                dist_ccknn = torch.sum(index_selected_diff_camid * dist[0], dim=1) / kc

                # Filter negative value before compensation
                intra_cam_comp_dist = torch.where(
                    (dist_ccknn - dist_icknn) > 0, (dist_ccknn - dist_icknn), torch.zeros(dist_ccknn.size())
                )
                intra_cam_comp_dist = intra_cam_comp_dist.unsqueeze(-1).expand(
                    intra_cam_comp_dist.size(0), intra_cam_comp_dist.size(0)
                ).float()

                # hyper-parameter of intra-camera compensation distance
                lambda_iccd = 0.9
                # hyper-parameter of cross-camera encouragement distance
                lambda_cced = 0.0
                # option of handling values greater than one after adding iccd,
                # 0: no operation; 1: filter; 2: cut-off
                gto_val_opt = 0
                # option of handling negative values after subtracting cced,
                # 0: no operation; 1: filter; 2: cut-off; 3: avoid
                nega_val_opt = 1

                # Push intra-camera pairs away by adding intra-camera compensation distance
                if option == 3 or option == 5:
                    # Filter values greater than 1.0
                    if gto_val_opt == 1:
                        dist[0] = torch.where(
                            (dist[0] + lambda_iccd * camids_sim * intra_cam_comp_dist) > 1.0,
                            dist[0],
                            (dist[0] + lambda_iccd * camids_sim * intra_cam_comp_dist)
                        )
                    # Cut off values greater than 1.0
                    elif gto_val_opt == 2:
                        dist[0] = torch.where(
                            (dist[0] + lambda_iccd * camids_sim * intra_cam_comp_dist) > 1.0,
                            0.99 * torch.ones(dist[0].size()),
                            (dist[0] + lambda_iccd * camids_sim * intra_cam_comp_dist)
                        )
                    # Without handling values greater than 1.0
                    else:
                        dist[0] = (
                                dist[0] + lambda_iccd * camids_sim * intra_cam_comp_dist
                        )

                # Pull cross-camera pairs closer by adding cross-camera encouragement distance
                # (Subtracting intra-camera compensation distance)
                if option == 4 or option == 5:
                    # Filter negative values
                    if nega_val_opt == 1:
                        dist[0] = torch.where(
                            (dist[0] - lambda_cced * (1.0 - camids_sim) * intra_cam_comp_dist) < 0.0,
                            dist[0],
                            (dist[0] - lambda_cced * (1.0 - camids_sim) * intra_cam_comp_dist)
                        )

                    # Cut off negative values
                    elif nega_val_opt == 2:
                        dist[0] = torch.where(
                            (dist[0] - lambda_cced * (1.0 - camids_sim) * intra_cam_comp_dist) < 0.0,
                            0.01 * torch.zeros(dist[0].size()),
                            (dist[0] - lambda_cced * (1.0 - camids_sim) * intra_cam_comp_dist)
                        )

                    # Avoid negative values by adaptive compensation
                    elif nega_val_opt == 3:
                        dist[0] = torch.where(
                            (dist[0] - lambda_cced * (1.0 - camids_sim) * intra_cam_comp_dist) < 0.0,
                            (1.0 - lambda_cced) * (1.0 - camids_sim) * dist[0],
                            (dist[0] - lambda_cced * (1.0 - camids_sim) * intra_cam_comp_dist)
                        )

                    # Without filtering or cutting off negative values
                    else:
                        dist[0] = (
                                dist[0] - lambda_cced * (1.0 - camids_sim) * intra_cam_comp_dist
                        )

                dist[0] = dist[0].cpu().numpy()

                del mask, dist_same_camid, index_sorted_same_camid, index_selected_same_camid, dist_icknn, \
                    dist_diff_camid, index_sorted_diff_camid, index_selected_diff_camid, dist_ccknn
            else:
                print("Invalid fusion option, keep the original feature distance.")

            del camids_sim

        features = [f.cpu() for f in features]

        # clustering
        eps = cfg.TRAIN.PSEUDO_LABELS.eps
        if len(eps) == 1:
            print("adopt independent clustering for every global and local branches")
            labels = []
            centers = []
            num_classes = []
            for i in range(len(features)):
                # normal clustering
                label, center, num_class = label_generator_dbscan_single(
                    cfg, features[i], dist[i], eps[0]
                )
                labels.append(label)
                centers.append(center)
                num_classes.append(num_class)
            # for regularization term of target loss, return jaccard distance matrix
            return labels, centers, num_classes, indep_thres, dist
            # original code
            # return labels, centers, num_classes, indep_thres
        else:
            print("adopt independent clustering for every global and local branches and fusing the results")
            assert (
                len(eps) == 2
            ), "multiple eps values are required for the fusion of clustering results"

            # global and local eps differs
            labels_global, _, num_classes = label_generator_dbscan_single(
                cfg, features[0], dist[0], eps[0]
            )

            labels_local_1, _, _ = label_generator_dbscan_single(
                cfg, features[1], dist[1], eps[1]
            )
            outliers = 0
            for i, label in enumerate(labels_local_1):
                if label == -1:
                    labels_local_1[i] += outliers
                    outliers -= 1

            labels_local_2, _, _ = label_generator_dbscan_single(
                cfg, features[2], dist[2], eps[1]
            )
            outliers = 0
            for i, label in enumerate(labels_local_2):
                if label == -1:
                    labels_local_2[i] += outliers
                    outliers -= 1

            # label similarity matrix
            labels_local_1_sim = (
                labels_local_1.expand(N, N).eq(labels_local_1.expand(N, N).t()).float()
            )
            labels_local_2_sim = (
                labels_local_2.expand(N, N).eq(labels_local_2.expand(N, N).t()).float()
            )

            # compute label similarity for pseudo label quality measure
            # print("Before pseudo label optimization: ")
            # print_pseudo_labels_quality_measure(labels_global, ground_truth_labels)

            end = time.time()
            print("cluster results fusing......")
            outlier_remv = 0
            labels_global = to_numpy(labels_global)
            labels_local_1_sim = to_numpy(labels_local_1_sim)
            labels_local_2_sim = to_numpy(labels_local_2_sim)
            labels = labels_global
            for i, label_i in enumerate(labels_global):
                if i == 0: break
                if label_i < 0:
                    for j, label_j in enumerate(labels_global):
                        if (labels_local_1_sim[i][j] == 1 or labels_local_2_sim[i][j] == 1) \
                                and label_j >= 0:
                            labels[i] = label_j
                            outlier_remv += 1
                            break

            # cluster labels -> pseudo labels
            # compute cluster centers
            centers = [collections.defaultdict(list) for _ in range(len(features))]
            outliers = 0
            for i, label in enumerate(labels):
                if label == -1:
                    if not cfg.TRAIN.PSEUDO_LABELS.use_outliers:
                        continue
                    labels[i] = num_classes + outliers
                    outliers += 1
                for j in range(len(features)):
                    centers[j][labels[i]].append(features[j][i])
            num_classes += outliers
            for i in range(len(features)):
                centers[i] = [
                    torch.stack(centers[i][idx], dim=0).mean(0) for idx in sorted(centers[i].keys())
                ]
                centers[i] = torch.stack(centers[i], dim=0)
            labels = to_torch(labels).long()
            print("Cluster results fusion cost: {}s".format(time.time() - end),
                  "with %d outliers removed." %(outlier_remv))

            # compute label similarity for pseudo label quality measure
            # print("After pseudo label optimization: ")
            # print_pseudo_labels_quality_measure(labels, ground_truth_labels)

            del labels_global, labels_local_1, labels_local_2, \
                labels_local_1_sim, labels_local_2_sim
            return labels, centers, num_classes, indep_thres, dist

    # only use global branch for cluster, instead of independent clustering for every branches
    else:
        # compute distance matrix by features
        # (Optional) ready for global-local feature extraction
        if isinstance(features, list):
            # use global features to calculate distance matrix
            dist = build_dist(cfg.TRAIN.PSEUDO_LABELS, features[0], verbose=True)

            features = [f.cpu() for f in features]
        else:
            # original code
            dist = build_dist(cfg.TRAIN.PSEUDO_LABELS, features, verbose=True)

            features = features.cpu()

        # clustering
        eps = cfg.TRAIN.PSEUDO_LABELS.eps
        if len(eps) == 1:
            # normal clustering
            labels, centers, num_classes = label_generator_dbscan_single(
                cfg, features, dist, eps[0]
            )
            # for regularization term of target loss, return jaccard distance matrix
            return labels, centers, num_classes, indep_thres, dist
            # original code
            # return labels, centers, num_classes, indep_thres

        else:
            assert (
                len(eps) == 3
            ), "three eps values are required for the clustering reliability criterion"
            assert (
                not isinstance(features, list)
            ), "global-local features are not supported by the clustering reliability criterion so far"

            print("adopt the reliability criterion for filtering clusters")
            eps = sorted(eps)
            labels_tight, _, _ = label_generator_dbscan_single(cfg, features, dist, eps[0])
            labels_normal, _, num_classes = label_generator_dbscan_single(
                cfg, features, dist, eps[1]
            )
            labels_loose, _, _ = label_generator_dbscan_single(cfg, features, dist, eps[2])

            # compute R_indep and R_comp
            N = labels_normal.size(0)
            label_sim = (
                labels_normal.expand(N, N).eq(labels_normal.expand(N, N).t()).float()
            )
            label_sim_tight = (
                labels_tight.expand(N, N).eq(labels_tight.expand(N, N).t()).float()
            )
            label_sim_loose = (
                labels_loose.expand(N, N).eq(labels_loose.expand(N, N).t()).float()
            )

            R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(
                label_sim, label_sim_tight).sum(-1)
            R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(
                label_sim, label_sim_loose).sum(-1)
            assert (R_comp.min() >= 0) and (R_comp.max() <= 1)
            assert (R_indep.min() >= 0) and (R_indep.max() <= 1)

            cluster_R_comp, cluster_R_indep = (
                collections.defaultdict(list),
                collections.defaultdict(list),
            )
            cluster_img_num = collections.defaultdict(int)
            for comp, indep, label in zip(R_comp, R_indep, labels_normal):
                cluster_R_comp[label.item()].append(comp.item())
                cluster_R_indep[label.item()].append(indep.item())
                cluster_img_num[label.item()] += 1

            cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
            cluster_R_indep = [
                min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())
            ]
            cluster_R_indep_noins = [
                iou
                for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys()))
                if cluster_img_num[num] > 1
            ]
            if indep_thres is None:
                indep_thres = np.sort(cluster_R_indep_noins)[
                    min(
                        len(cluster_R_indep_noins) - 1,
                        np.round(len(cluster_R_indep_noins) * 0.9).astype("int"),
                    )
                ]

            labels_num = collections.defaultdict(int)
            for label in labels_normal:
                labels_num[label.item()] += 1

            centers = collections.defaultdict(list)
            outliers = 0
            for i, label in enumerate(labels_normal):
                label = label.item()
                indep_score = cluster_R_indep[label]
                comp_score = R_comp[i]
                if label == -1:
                    assert not cfg.TRAIN.PSEUDO_LABELS.use_outliers, "exists a bug"
                    continue
                if (indep_score > indep_thres) or (
                    comp_score.item() > cluster_R_comp[label]
                ):
                    if labels_num[label] > 1:
                        labels_normal[i] = num_classes + outliers
                        outliers += 1
                        labels_num[label] -= 1
                        labels_num[labels_normal[i].item()] += 1

                centers[labels_normal[i].item()].append(features[i])

            num_classes += outliers
            assert len(centers.keys()) == num_classes

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]
            centers = torch.stack(centers, dim=0)

            return labels_normal, centers, num_classes, indep_thres


@torch.no_grad()
def print_pseudo_labels_quality_measure(pseudo_labels, ground_truth_labels):
    pseudo_labels = pseudo_labels.clone().detach()
    N = pseudo_labels.size(0)
    outliers = 0
    for i, label in enumerate(pseudo_labels):
        if label == -1:
            pseudo_labels[i] += outliers
            outliers -= 1
    labels_sim = (
        pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
    )
    labels_grt_sim = (
        ground_truth_labels.expand(N, N).eq(ground_truth_labels.expand(N, N).t()).float()
    )
    predicted_pos = int(labels_sim.cpu().sum())
    grt_pos = int(labels_grt_sim.cpu().sum())
    labels_grt_sim = labels_grt_sim.cuda()
    labels_grt_sim = torch.where(
        labels_grt_sim == 0, torch.full_like(labels_grt_sim, 2.0), labels_grt_sim
    ).cpu()
    true_pos = float(labels_grt_sim.eq(labels_sim).cpu().sum())
    true_pos_rate = true_pos / (N * N) * 100.0
    pairwise_acc = true_pos / predicted_pos * 100.0
    pairwise_recall = true_pos / grt_pos * 100.0
    print("Pseudo labels pair-wise true positive rate: %f%%; accuracy rate: %f%%; recall rate: %f%%"
          % (true_pos_rate, pairwise_acc, pairwise_recall))
    del labels_grt_sim, labels_sim


