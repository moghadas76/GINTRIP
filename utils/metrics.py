import torch
import torch.nn as nn
import sys, os
import os.path as osp
from typing import List, Union
import numpy as np
from torch_geometric.data.data import Data
from torch_geometric.nn import MessagePassing
from torch import Tensor


def control_sparsity(mask: torch.Tensor, sparsity: float=None):
    r"""
    Transform the mask where top 1 - sparsity values are set to inf.
    Args:
        mask (torch.Tensor): Mask that need to transform.
        sparsity (float): Sparsity we need to control i.e. 0.7, 0.5 (Default: :obj:`None`).
    :rtype: torch.Tensor
    """
    if sparsity is None:
        sparsity = 0.7

    _, indices = torch.sort(mask, descending=True)
    mask_len = mask.shape[0]
    split_point = int((1 - sparsity) * mask_len)
    important_indices = indices[: split_point]
    unimportant_indices = indices[split_point:]
    trans_mask = mask.clone()
    trans_mask[important_indices] = float('inf')
    trans_mask[unimportant_indices] = - float('inf')

    return trans_mask


def fidelity(ori_probs: torch.Tensor, unimportant_probs: torch.Tensor) -> float:
    r"""
    Return the Fidelity+ value according to collected data.

    Args:
        ori_probs (torch.Tensor): It is a vector providing original probabilities for Fidelity+ computation.
        unimportant_probs (torch.Tensor): It is a vector providing probabilities without important features
            for Fidelity+ computation.

    :rtype: float

    .. note::
        Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
        <https://arxiv.org/abs/2012.15445>`_ for details.
    """

    # drop_probability = ori_probs - unimportant_probs
    loss_fuc = nn.L1Loss()
    drop_probability = loss_fuc(ori_probs , unimportant_probs)

    return drop_probability.item()

def fidelity_abs(ori_preds: torch.Tensor, unimportant_preds: torch.Tensor, trues: torch.Tensor) -> float:
    r"""
    Return the Fidelity+ value according to collected data.

    Args:
        ori_probs (torch.Tensor): It is a vector providing original probabilities for Fidelity+ computation.
        unimportant_probs (torch.Tensor): It is a vector providing probabilities without important features
            for Fidelity+ computation.

    :rtype: float

    .. note::
        Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
        <https://arxiv.org/abs/2012.15445>`_ for details.
    """

    # drop_probability = ori_probs - unimportant_probs
    loss_fuc = nn.L1Loss()
    print(ori_preds.shape, trues.shape)
    ori_mae = loss_fuc(ori_preds, trues) # small
    unimportant_mae = loss_fuc(unimportant_preds, trues) # big
    drop_probability = unimportant_mae - ori_mae

    return drop_probability.item()


def fidelity_inv(ori_probs: torch.Tensor, important_probs: torch.Tensor) -> float:
    r"""
    Return the Fidelity- value according to collected data.

    Args:
        ori_probs (torch.Tensor): It is a vector providing original probabilities for Fidelity- computation.
        important_probs (torch.Tensor): It is a vector providing probabilities with only important features
            for Fidelity- computation.

    :rtype: float

    .. note::
        Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
        <https://arxiv.org/abs/2012.15445>`_ for details.
    """

    # drop_probability = ori_probs - important_probs
    loss_fuc = nn.L1Loss()
    drop_probability = loss_fuc(ori_probs , important_probs)

    return drop_probability.item()



class XCollector:
    r"""
    XCollector is a data collector which takes processed related prediction probabilities to calculate Fidelity+
    and Fidelity-.

    Args:
        sparsity (float): The Sparsity is use to transform the soft mask to a hard one.

    .. note::
        For more examples, see `benchmarks/xgraph
        <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.

    """

    def __init__(self, sparsity=None):
        self.__related_preds, self.__targets = \
            {
                'zero': [],
                'masked': [],
                'maskout': [],
                'origin': [],
                'sparsity': [],
                'accuracy': [],
                'stability': [], 
                'trues': []
             }, []
        self.masks: Union[List, List[List[Tensor]]] = []

        self.__sparsity = sparsity
        self.__fidelity, self.__fidelity_inv, self.__accuracy, self.__stability = None, None, None, None
        self.__fidelity_abs = None
        self.__score = None

    @property
    def targets(self) -> list:
        return self.__targets

    def new(self):
        r"""
        Clear class members.
        """
        self.__related_preds, self.__targets = \
            {
                'zero': [],
                'masked': [],
                'maskout': [],
                'origin': [],
                'sparsity': [],
                'accuracy': [],
                'stability': [], 
                'trues': []
             }, []
        self.masks: Union[List, List[List[Tensor]]] = []
        self.__fidelity, self.__fidelity_inv, self.__accuracy, self.__stability = None, None, None, None
        self.__fidelity_abs = None

    def collect_data(self,
                     masks: List[Tensor],
                     related_preds: dir,
                     label: int = 0) -> None:
        r"""
        The function is used to collect related data. After collection, we can call fidelity, fidelity_inv, sparsity
        to calculate their values.

        Args:
            masks (list): It is a list of edge-level explanation for each class.
            related_preds (list): It is a list of dictionary for each class where each dictionary
            includes 4 type predicted probabilities and sparsity.
            label (int): The ground truth label. (default: 0)
        """

        if self.__fidelity is not None or self.__fidelity_inv is not None \
                or self.__accuracy is not None or self.__stability is not None or self.__fidelity_abs is not None:
            self.__fidelity, self.__fidelity_inv, self.__accuracy, self.__stability = None, None, None, None
            self.__fidelity_abs = None
            print(f'#W#Called collect_data() after calculate explainable metrics.')

        if not np.isnan(label):
            for key, value in related_preds[label].items():
                self.__related_preds[key].append(value)
            for key in self.__related_preds.keys():
                if key not in related_preds[0].keys():
                    self.__related_preds[key].append(None)
            self.__targets.append(label)
            self.masks.append(masks)

    @property
    def fidelity(self):
        r"""
        Return the Fidelity+ value according to collected data.

        .. note::
            Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
            <https://arxiv.org/abs/2012.15445>`_ for details.
        """
        if self.__fidelity is not None:
            return self.__fidelity
        else:
            if None in self.__related_preds['maskout'] or None in self.__related_preds['origin']:
                return None
            else:
                # mask_out_preds, one_mask_preds = \
                #     torch.tensor(self.__related_preds['maskout']), torch.tensor(self.__related_preds['origin'])
                mask_out_preds = torch.stack(self.__related_preds['maskout'], dim=0)
                one_mask_preds = torch.stack(self.__related_preds['origin'], dim=0)
                self.__fidelity = fidelity(one_mask_preds, mask_out_preds)
                return self.__fidelity

    @property
    def fidelity_abs(self):
        r"""
        Return the Fidelity+ value according to collected data.

        .. note::
            Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
            <https://arxiv.org/abs/2012.15445>`_ for details.
        """
        if self.__fidelity_abs is not None:
            return self.__fidelity_abs
        else:
            if None in self.__related_preds['maskout'] or None in self.__related_preds['origin']:
                return None
            else:
                # mask_out_preds, one_mask_preds = \
                #     torch.tensor(self.__related_preds['maskout']), torch.tensor(self.__related_preds['origin'])
                mask_out_preds = torch.stack(self.__related_preds['maskout'], dim=0)
                one_mask_preds = torch.stack(self.__related_preds['origin'], dim=0)
                trues = torch.stack(self.__related_preds['trues'], dim = 0)
                self.__fidelity_abs = fidelity_abs(one_mask_preds, mask_out_preds, trues)
                return self.__fidelity_abs

    @property
    def fidelity_inv(self):
        r"""
        Return the Fidelity- value according to collected data.

        .. note::
            Please refer to `Explainability in Graph Neural Networks: A Taxonomic Survey
            <https://arxiv.org/abs/2012.15445>`_ for details.
        """
        if self.__fidelity_inv is not None:
            return self.__fidelity_inv
        else:
            if None in self.__related_preds['masked'] or None in self.__related_preds['origin']:
                return None
            else:
                # masked_preds, one_mask_preds = \
                #     torch.tensor(self.__related_preds['masked']), torch.tensor(self.__related_preds['origin'])
                masked_preds = torch.stack(self.__related_preds['masked'], dim=0)
                one_mask_preds = torch.stack(self.__related_preds['origin'], dim=0)

                self.__fidelity_inv = fidelity_inv(one_mask_preds, masked_preds)
                return self.__fidelity_inv

    @property
    def sparsity(self):
        r"""
        Return the Sparsity value.
        """
        if self.__sparsity is not None:
            return self.__sparsity
        else:
            if None in self.__related_preds['sparsity']:
                return None
            else:
                
                return torch.tensor(self.__related_preds['sparsity']).mean().item()

    @property
    def accuracy(self):
        r"""Return the accuracy for datasets with motif ground-truth"""
        if self.__accuracy is not None:
            return self.__accuracy
        else:
            if None in self.__related_preds['accuracy']:
                return torch.tensor([acc for acc in self.__related_preds['accuracy']
                                     if acc is not None]).mean().item()
            else:
                return torch.tensor(self.__related_preds['accuracy']).mean().item()

    @property
    def stability(self):
        r"""Return the accuracy for datasets with motif ground-truth"""
        if self.__stability is not None:
            return self.__stability
        else:
            if None in self.__related_preds['stability']:
                return torch.tensor([stability for stability in self.__related_preds['stability']
                                     if stability is not None]).mean().item()
            else:
                return torch.tensor(self.__related_preds['stability']).mean().item()
