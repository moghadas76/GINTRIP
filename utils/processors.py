import torch
import torch.nn as nn
from typing import List, Union
from torch import Tensor
import numpy as np
from torch_geometric.data.data import Data
from torch_geometric.nn import MessagePassing

from utils.metrics import XCollector


class ExplanationProcessor(nn.Module):
    r"""
    Explanation Processor is edge mask explanation processor which can handle sparsity control and use
    data collector automatically.

    Args:
        model (torch.nn.Module): The target model prepared to explain.
        device (torch.device): Specify running device: CPU or CUDA.

    """

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__()
        self.edge_mask = None
        self.model = model
        self.device = device
        self.mp_layers = [module for module in self.model.modules() if isinstance(module, MessagePassing)]
        self.num_layers = len(self.mp_layers)

    class connect_mask(object):

        def __init__(self, cls):
            self.cls = cls

        def __enter__(self):

            self.cls.edge_mask = [nn.Parameter(torch.randn(self.cls.x_batch_size * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)] if hasattr(self.cls, 'x_batch_size') else \
                                 [nn.Parameter(torch.randn(1 * (self.cls.num_edges + self.cls.num_nodes))) for _ in
                             range(self.cls.num_layers)]

            for idx, module in enumerate(self.cls.mp_layers):
                module._explain = True
                module.__edge_mask__ = self.cls.edge_mask[idx]

        def __exit__(self, *args):
            for idx, module in enumerate(self.cls.mp_layers):
                module._explain = False

    def eval_related_pred(self, x: torch.Tensor, edge_index: torch.Tensor, masks: List[torch.Tensor], **kwargs):

        node_idx = kwargs.get('node_idx')
        node_idx = 0 if node_idx is None else node_idx # graph level: 0, node level: node_idx

        related_preds = []

        for label, mask in enumerate(masks):
            # origin pred
            for edge_mask in self.edge_mask:
                edge_mask.data = float('inf') * torch.ones(mask.size(), device=self.device)
            ori_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            for edge_mask in self.edge_mask:
                edge_mask.data = mask
            masked_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # mask out important elements for fidelity calculation
            for edge_mask in self.edge_mask:
                edge_mask.data = - mask
            maskout_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # zero_mask
            for edge_mask in self.edge_mask:
                edge_mask.data = - float('inf') * torch.ones(mask.size(), device=self.device)
            zero_mask_pred = self.model(x=x, edge_index=edge_index, **kwargs)

            # Store related predictions for further evaluation.
            related_preds.append({'zero': zero_mask_pred[node_idx],
                                  'masked': masked_pred[node_idx],
                                  'maskout': maskout_pred[node_idx],
                                  'origin': ori_pred[node_idx]})

            # Adding proper activation function to the models' outputs.
            related_preds[label] = {key: pred.softmax(0)[label].item()
                                    for key, pred in related_preds[label].items()}

        return related_preds

    def forward(self, data: Data, masks: List[torch.Tensor], x_collector: XCollector, **kwargs):
        r"""
        Please refer to the main function in `metric.py`.
        """
        breakpoint()
        data.to(self.device)
        node_idx = kwargs.get('node_idx')
        y_idx = 0 if node_idx is None else node_idx

        assert not torch.isnan(data.y[y_idx].squeeze())

        self.num_edges = data.edge_index.shape[1]
        self.num_nodes = data.x.shape[0]

        with torch.no_grad():
            with self.connect_mask(self):
                related_preds = self.eval_related_pred(data.x, data.edge_index, masks, **kwargs)

        x_collector.collect_data(masks,
                                 related_preds,
                                 data.y[y_idx].squeeze().long().item())
