import torch
import torch.nn.functional as F
from torch.autograd import Variable, Function

from torch import nn
from torch.nn import init

from ...utils.dist_utils import all_gather_tensor


#  based on invariance v2 ----------------------------------------------------------------------------

# instance-level memory
class InstanceMemory(Function):
    '''def __init__(self, im, momentum=0.01):
        super(InstanceMemory, self).__init__()
        self.im = im
        self.momentum = momentum

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.im.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.im)
        for x, y in zip(inputs, targets):
            self.im[y] = self.momentum * self.im[y] + (1. - self.momentum) * x
            self.im[y] /= self.im[y].norm()
        return grad_inputs, None'''

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        outputs = inputs.mm(ctx.features.t())
        all_inputs = all_gather_tensor(inputs)
        all_targets = all_gather_tensor(targets)
        ctx.save_for_backward(all_inputs, all_targets)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()
        return grad_inputs, None, None, None


def im(inputs, indexes, features, momentum=0.5):
    return InstanceMemory.apply(
        inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device)
    )


# Instance memory loss
class InstanceMemoryLoss(nn.Module):
    def __init__(self, num_features, num_classes, temp=0.05, momentum=0.01, knn=6):
        super(InstanceMemoryLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features  # ------------------------  + 2048  ------------------------------
        self.num_classes = num_classes
        self.momentum = momentum  # Memory update rate
        self.temp = temp  # Temperature fact
        self.knn = knn  # Knn for neighborhood invariance

        # Instance memory
        self.register_buffer("im", torch.zeros(num_classes, num_features))
        # self.im = nn.Parameter(torch.zeros(self.num_classes, self.num_features))

    @torch.no_grad()
    def _update_feature(self, features):
        features = F.normalize(features, p=2, dim=1)
        self.im.data.copy_(features.float().to(self.im.device))

    def forward(self, inputs, targets, dist, epoch):
        inputs = inputs["feat"]
        # concat global and local feature for instance memory loss
        if isinstance(inputs, list):
            inputs = torch.cat([f for f in inputs], dim=1).to(inputs[0].device)
        inputs = F.normalize(inputs, p=2, dim=1)
        inputs = torch.split(inputs, inputs.size(0) // 2, dim=0)
        inputs = inputs[1]

        targets = torch.split(targets, targets.size(0) // 2, dim=0)
        targets = targets[1]

        # dist = torch.Tensor(dist).to(inputs.device)

        # momentum = self.momentum * epoch
        momentum = self.momentum
        tgt_feat = im(inputs, targets, self.im, momentum)
        # original code
        # tgt_feat = InstanceMemory(self.im, momentum=momentum)(inputs, targets)
        tgt_feat /= self.temp

        if self.knn > 0 and epoch > 49:  # -----------------------------------------------------------------
            loss = self.smooth_loss_with_levels(tgt_feat, dist,
                                                targets)  # --------------------------------------------------
        elif self.knn > 0 and epoch > 1:
            loss = self.smooth_loss(tgt_feat, targets)
        else:
            loss = F.cross_entropy(tgt_feat, targets)
        return loss

    def smooth_loss(self, inputs, targets):
        targets = self.smooth_hot(inputs.detach().clone(), targets.detach().clone(), self.knn)
        outputs = F.log_softmax(inputs, dim=1)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def smooth_hot(self, inputs, targets, k=6):
        # Sort
        _, index_sorted = torch.sort(inputs, dim=1, descending=True)

        ones_mat = torch.ones(targets.size(0), k).to(self.device)
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.zeros(inputs.size()).to(self.device)

        weights = F.softmax(ones_mat, dim=1)
        targets_onehot.scatter_(1, index_sorted[:, 0:k], ones_mat * weights)
        targets_onehot.scatter_(1, targets, float(1))

        return targets_onehot

    def smooth_loss_with_levels(self, inputs, dist, targets):  # ---------------------------------------------------
        targets = self.smooth_hot_with_levels(inputs.detach().clone(), targets.detach().clone(), dist.detach().clone(),
                                              self.knn)  # ---------------------------------------------------
        outputs = F.log_softmax(inputs, dim=1)
        loss = - (targets * outputs)
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=0)
        return loss

    def smooth_hot_with_levels(self, inputs, targets, dist,
                               k=6):  # ------------------------------------------------------------
        # cosine distance sort
        # start = time.time()

        _, index_sorted = torch.sort(inputs, dim=1, descending=True)

        ones_mat = torch.ones(targets.size(0), k).to(self.device)
        targets = torch.unsqueeze(targets, 1)
        targets_onehot = torch.zeros(inputs.size()).to(self.device)

        weights = F.softmax(ones_mat, dim=1)  # assign 1/k
        targets_onehot.scatter_(1, index_sorted[:, 0:k], ones_mat * weights)
        targets_onehot.scatter_(1, targets, float(1))

        # k-reciprocal encoding sort
        inputs_rerank = dist

        # Sort
        _, index_sorted_rerank = torch.sort(inputs_rerank, dim=1,
                                            descending=False)  # ----------------------------ascending order -----------------------------------------

        ones_mat_rerank = torch.ones(ones_mat.size(0), k).to(self.device)
        targets_onehot_rerank = torch.zeros(inputs_rerank.size()).to(self.device)

        weights_rerank = F.softmax(ones_mat_rerank, dim=1)  # assign 1/k
        targets_onehot_rerank.scatter_(1, index_sorted_rerank[:, 0:k],
                                       ones_mat_rerank * weights_rerank)  # ---------all k-reciprocal neighborhoods are assigned weight 1/k ------------------
        # targets_onehot_rerank.scatter_(1, targets, float(1))

        targets_onehot = targets_onehot + targets_onehot_rerank
        # targets_onehot = torch.clamp(targets_onehot, float(0), float(1))

        return targets_onehot




