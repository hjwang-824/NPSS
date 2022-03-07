# version description: joint Global-Local learning with instance-memory loss (GL)
# need to maintain and update a jaccard distance matrix during every training epoch
# Fusion Clustering results (FC)
# add Soft losses for global and local branches (S)
# with cross-domain Mixup (M)
# add inter-instance soft supervision (I)

import torch
import torch.nn as nn

__all__ = ["CrossEntropyLoss", "SoftEntropyLoss", "InterInstanceSoftEntropyLoss"]


class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
    num_classes (int): number of classes.
    epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

        self.logsoftmax = nn.LogSoftmax(dim=1)
        assert self.num_classes > 0

    def forward(self, results, targets, is_mixup=False, mixup_lmd=None):
        """
        Args:
        inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        targets: ground truth labels with shape (num_classes)
        """

        inputs = results["prob"]

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        if is_mixup:
            batch_size_per_domain = inputs.size(0) // 2
            #targets[:batch_size_per_domain] = mixup_lmd * targets[:batch_size_per_domain] + \
            #                                 (1 - mixup_lmd) * targets[batch_size_per_domain:]
            #targets[batch_size_per_domain:] = mixup_lmd * targets[batch_size_per_domain:] + \
            #                                     (1 - mixup_lmd) * targets[:batch_size_per_domain]
            """
            temp = targets
            targets[:batch_size_per_domain] = mixup_lmd * temp[:batch_size_per_domain] + \
                                             (1 - mixup_lmd) * temp[batch_size_per_domain:]
            targets[batch_size_per_domain:] = mixup_lmd * temp[batch_size_per_domain:] + \
                                                 (1 - mixup_lmd) * temp[:batch_size_per_domain]
            """

            # SourceMix + Source + Target + TargetMix
            batch_size_mix = batch_size_per_domain // 2
            temp = targets
            targets[:batch_size_mix] = \
                mixup_lmd * temp[:batch_size_mix] + \
                (1 - mixup_lmd) * temp[batch_size_mix*2:batch_size_mix*3]
            targets[batch_size_mix*3:] = \
                mixup_lmd * temp[batch_size_mix*3:] + \
                (1 - mixup_lmd) * temp[batch_size_mix:batch_size_mix*2]
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        loss = (-targets * log_probs).mean(0).sum()
        return loss


class SoftEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, results, results_supvs, results_supvs2=None):
        assert results_supvs is not None

        inputs = results["prob"]
        if results_supvs2 is not None:
            fusion_option = 0
            if fusion_option == 1:
                lmd_self = 0.5
                targets = lmd_self * inputs \
                          + (1 - lmd_self) * 0.5 * (results_supvs["prob"] + results_supvs2["prob"])
            else:
                targets = 0.5 * (results_supvs["prob"] + results_supvs2["prob"])
        else:
            targets = results_supvs["prob"]

        log_probs = self.logsoftmax(inputs)
        loss = (-self.softmax(targets).detach() * log_probs).mean(0).sum()
        return loss


# Inter instance (belong to the same identity) soft entropy loss
class InterInstanceSoftEntropyLoss(nn.Module):
    def __init__(self):
        super(InterInstanceSoftEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, results, hard_targets):
        assert hard_targets is not None

        inputs = results["prob"]

        N = hard_targets.numel()
        ids_eq = hard_targets.expand(N, N).eq(hard_targets.expand(N, N).t()).float() \
                 - torch.eye(N, N).to(hard_targets.device)

        inter_instance_per_id = torch.div(torch.sum(ids_eq), float(N))
        # print(inter_instance_per_id)

        fusion_option = 0
        if fusion_option == 1:
            lmd_self = 0.5
            targets = lmd_self * inputs \
                      + (1 - lmd_self) * torch.mm(torch.div(ids_eq, float(inter_instance_per_id)), inputs)
        else:
            targets = torch.mm(torch.div(ids_eq, float(inter_instance_per_id)), inputs)

        log_probs = self.logsoftmax(inputs)
        loss = (-self.softmax(targets).detach() * log_probs).mean(0).sum()
        return loss

