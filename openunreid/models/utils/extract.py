# version description: joint global-local learning with instance-memory loss
# need to maintain and update a jaccard distance matrix during every training epoch
# fusion the clustering results of global and local branches

import time
from collections import OrderedDict

import torch
import torch.nn.functional as F

from ...utils.dist_utils import all_gather_tensor, get_dist_info, synchronize
from ...utils.meters import Meters


@torch.no_grad()
def extract_features(
    model,  # model used for extracting
    data_loader,  # loading data
    dataset,  # dataset with file paths, etc
    cuda=True,  # extract on GPU
    normalize=True,  # normalize feature
    with_path=False,  # return a dict {path:feat} if True, otherwise, return only feat (Tensor)  # noqa
    print_freq=10,  # log print frequence
    save_memory=False,  # gather features from different GPUs all together or in sequence, only for distributed  # noqa
    for_testing=True,
    return_labels=False,
    return_camids=False,
    prefix="Extract: ",
):

    progress = Meters({"Time": ":.3f", "Data": ":.3f"}, len(data_loader), prefix=prefix)

    rank, world_size, is_dist = get_dist_info()
    features = []
    labels = []
    camids = []

    model.eval()
    data_iter = iter(data_loader)

    end = time.time()
    for i in range(len(data_loader)):
        data = next(data_iter)
        progress.update({"Data": time.time() - end})

        images = data["img"]
        if cuda:
            images = images.cuda()

        # compute output
        outputs = model(images)

        if isinstance(outputs, list) and for_testing:
            outputs = torch.cat(outputs, dim=1)

        if normalize:
            if isinstance(outputs, list):
                outputs = [F.normalize(out, p=2, dim=-1) for out in outputs]
                # outputs = F.normalize(outputs[1], p=2, dim=-1)
            else: # fix an original bug
                outputs = F.normalize(outputs, p=2, dim=-1)

        if isinstance(outputs, list):
            outputs = torch.cat(outputs, dim=1).data.cpu()
        else:
            outputs = outputs.data.cpu()

        features.append(outputs)

        if return_labels:
            labs = data["id"]
            if cuda:
                labs = labs.cuda()
            labels.append(labs)
        
        if return_camids:
            camid = data["cid"]
            if cuda:
                camid = camid.cuda()
            camids.append(camid)
            

        # measure elapsed time
        progress.update({"Time": time.time() - end})
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    synchronize()

    if is_dist and cuda:
        # distributed: gather features from all GPUs
        features = torch.cat(features)
        all_features = all_gather_tensor(features.cuda(), save_memory=save_memory)
        all_features = all_features.cpu()[: len(dataset)]
        if return_labels:
            labels = torch.cat(labels)
            all_labels = all_gather_tensor(labels, save_memory=save_memory)
            all_labels = all_labels.cpu()[: len(dataset)]
        if return_camids:
            camids = torch.cat(camids)
            all_camids = all_gather_tensor(camids, save_memory=save_memory)
            all_camids = all_camids.cpu()[: len(dataset)]    
    else:
        # no distributed, no gather
        all_features = torch.cat(features, dim=0)[: len(dataset)]
        if return_labels:
            all_labels = torch.cat(labels, dim=0)[: len(dataset)]
        if return_camids:
            all_camids = torch.cat(camids, dim=0)[: len(dataset)]

    if not with_path:
        if return_labels and not return_camids:
            return all_features, all_labels
        elif return_labels and return_camids:
            return all_features, all_labels, all_camids
        else:
            return all_features

    features_dict = OrderedDict()
    for fname, feat in zip(dataset, all_features):
        features_dict[fname[0]] = feat

    return features_dict

