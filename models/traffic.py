from __future__ import division
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from itertools import accumulate
from torch_geometric.utils import dense_to_sparse
import math
from .GAT import GATConvExp


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()


class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,nvwl->ncwl', (x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        """
        X => torch.Size([64, 32, 207, 13])
        """
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class dy_mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep + 1) * c_in, c_out)
        self.mlp2 = linear((gdep + 1) * c_in, c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in, c_in)
        self.lin2 = linear(c_in, c_in)

    def forward(self, x):
        # adj = adj + torch.eye(adj.size(0)).to(x.device)
        # d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2, 1), x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2, 1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj0)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho1 = self.mlp1(ho)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1 + ho2


class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        self.tconv = nn.Conv2d(cin, cout, (1, 7), dilation=(1, dilation_factor))

    def forward(self, input):
        x = self.tconv(input)
        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class graph_constructor(nn.Module): #TODO: mohammad siad: this class is just from a traffic spesific repository or paper. does it suiutable for other datasets like weather?
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj


class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:, idx, :], self.bias[:, idx, :], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


# class gtnet(nn.Module):
#     def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None,
#                  dropout=0.3, subgraph_size=20, node_dim=40, dilation_exponential=1, conv_channels=32,
#                  residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12,
#                  layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
#         super(gtnet, self).__init__()
#         self.gcn_true = gcn_true
#         self.buildA_true = buildA_true
#         self.num_nodes = num_nodes
#         self.dropout = dropout
#         self.predefined_A = predefined_A
#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.residual_convs = nn.ModuleList()
#         self.skip_convs = nn.ModuleList()
#         self.gconv1 = nn.ModuleList()
#         self.gconv2 = nn.ModuleList()
#         self.norm = nn.ModuleList()
#         self.start_conv = nn.Conv2d(in_channels=in_dim,
#                                     out_channels=residual_channels,
#                                     kernel_size=(1, 1))
#         self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
#                                     static_feat=static_feat)

#         self.seq_length = seq_length
#         kernel_size = 7
#         if dilation_exponential > 1:
#             self.receptive_field = int(
#                 1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
#         else:
#             self.receptive_field = layers * (kernel_size - 1) + 1

#         for i in range(1):
#             if dilation_exponential > 1:
#                 rf_size_i = int(
#                     1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
#             else:
#                 rf_size_i = i * layers * (kernel_size - 1) + 1
#             new_dilation = 1
#             for j in range(1, layers + 1):
#                 if dilation_exponential > 1:
#                     rf_size_j = int(
#                         rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
#                 else:
#                     rf_size_j = rf_size_i + j * (kernel_size - 1)

#                 self.filter_convs.append(
#                     dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
#                 self.gate_convs.append(
#                     dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
#                 self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
#                                                      out_channels=residual_channels,
#                                                      kernel_size=(1, 1)))
#                 if self.seq_length > self.receptive_field:
#                     self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
#                                                      out_channels=skip_channels,
#                                                      kernel_size=(1, self.seq_length - rf_size_j + 1)))
#                 else:
#                     self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
#                                                      out_channels=skip_channels,
#                                                      kernel_size=(1, self.receptive_field - rf_size_j + 1)))

#                 if self.gcn_true:
#                     self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
#                     self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

#                 if self.seq_length > self.receptive_field:
#                     self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
#                                                elementwise_affine=layer_norm_affline))
#                 else:
#                     self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
#                                                elementwise_affine=layer_norm_affline))

#                 new_dilation *= dilation_exponential

#         self.layers = layers
#         self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
#                                     out_channels=end_channels,
#                                     kernel_size=(1, 1),
#                                     bias=True)
#         self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
#                                     out_channels=out_dim,
#                                     kernel_size=(1, 1),
#                                     bias=True)
#         if self.seq_length > self.receptive_field:
#             self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
#                                    bias=True)
#             self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
#                                    kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

#         else:
#             self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
#                                    kernel_size=(1, self.receptive_field), bias=True)
#             self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
#                                    bias=True)

#         self.idx = torch.arange(self.num_nodes).to(device)

#     def forward(self, input, idx=None):
#         seq_len = input.size(3)
#         assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

#         if self.seq_length < self.receptive_field:
#             input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

#         if self.gcn_true:
#             if self.buildA_true:
#                 if idx is None:
#                     adp = self.gc(self.idx)
#                 else:
#                     adp = self.gc(idx)
#             else:
#                 adp = self.predefined_A

#         x = self.start_conv(input)
#         skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
#         for i in range(self.layers):
#             residual = x
#             filter = self.filter_convs[i](x)
#             filter = torch.tanh(filter)
#             gate = self.gate_convs[i](x)
#             gate = torch.sigmoid(gate)
#             x = filter * gate
#             x = F.dropout(x, self.dropout, training=self.training)
#             s = x
#             s = self.skip_convs[i](s)
#             skip = s + skip
#             if self.gcn_true:
#                 x = self.gconv1[i](x, adp) + self.gconv2[i](x, adp.transpose(1, 0))
#             else:
#                 x = self.residual_convs[i](x)

#             x = x + residual[:, :, :, -x.size(3):]
#             if idx is None:
#                 x = self.norm[i](x, self.idx)
#             else:
#                 x = self.norm[i](x, idx)

#         skip = self.skipE(x) + skip
#         x = F.relu(skip)
#         x = F.relu(self.end_conv_1(x))
#         x = self.end_conv_2(x)
#         return x


class ProtoGtnet(nn.Module):
    """
    The base method came from the repo: 
    - https://github.com/nnzhan/MTGNN
    """

    def __init__(self,
                 gcn_true,
                 buildA_true,
                 gcn_depth,
                 num_nodes,
                 device,
                 predefined_A=None,
                 static_feat=None,
                 dropout=0.3,
                 subgraph_size=20,
                 node_dim=40,
                 dilation_exponential=1,
                 conv_channels=32,
                 residual_channels=32,
                 skip_channels=64,
                 end_channels=128,
                 seq_length=12,
                 in_dim=2,
                 out_dim=12,
                 layers=3,
                 propalpha=0.05,
                 tanhalpha=3,
                 layer_norm_affline=True,
                 top_k=6,
                 is_xai: dict=None,
                 addaptive_coef=False,
                 mi_loss=False,
                 graph_reg=False,
                 mean_pool=False,
                 ):
        super(ProtoGtnet, self).__init__()
        if is_xai is None:
            is_xai = {"xai": True, "gsub": True, "proto": True}
        self.is_xai = is_xai
        self.mi_loss = mi_loss
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.addaptive_coef = addaptive_coef
        ##########################################
        self.graph_reg=graph_reg
        self.mean_pool=mean_pool
        ##########################################
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha,
                                    static_feat=static_feat)

        self.seq_length = seq_length
        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1))
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1
            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size - 1) * (dilation_exponential ** j - 1) / (dilation_exponential - 1))
                else:
                    rf_size_j = rf_size_i + j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(
                    dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.seq_length - rf_size_j + 1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                     out_channels=skip_channels,
                                                     kernel_size=(1, self.receptive_field - rf_size_j + 1)))

                if self.gcn_true:
                    self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                else:
                    #TODO: Test with concat = False
                    self.gconv1.append(GATConvExp(conv_channels, 32, heads=4, concat=True))
                    self.gconv2.append(GATConvExp(conv_channels, 32, heads=4, concat=True))
                if self.seq_length > self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),
                                               elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length),
                                   bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels,
                                   kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels,
                                   kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1),
                                   bias=True)

        ### Prototype Section
        if self.is_xai["xai"]:
            if self.graph_reg:
                self.fc_cat = nn.Linear(64, 2)
                self.fc_out = nn.Linear((64) * 2, 64)

                def encode_one_hot(labels):
                # reference code https://github.com/chaoshangcs/GTS/blob/8ed45ff1476639f78c382ff09ecca8e60523e7ce/model/pytorch/model.py#L149
                    classes = set(labels)
                    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
                    labels_one_hot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
                    return labels_one_hot

                self.rel_rec = torch.FloatTensor(np.array(encode_one_hot(np.where(np.ones((self.num_nodes, self.num_nodes)))[0]), dtype=np.float32))
                self.rel_send = torch.FloatTensor(np.array(encode_one_hot(np.where(np.ones((self.num_nodes, self.num_nodes)))[1]), dtype=np.float32))
            self.fully_connected_1 = torch.nn.Linear(64, 64)
            self.fully_connected_2 = torch.nn.Linear(64, 2)
            self.Softmax = nn.Softmax(dim=-1)
            self.output_dim = 2
            self.prototype_per_class = 7
            self.latent = end_channels
            self.prototype_shape = (self.output_dim * self.prototype_per_class, 64)
            self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
            self.prototype_predictor = nn.Linear(64, self.prototype_per_class * self.output_dim * 64, bias=False)
            self.loss_fn = torch.nn.MSELoss()
            self.epsilon = 1e-4
            if self.is_xai["proto"]:
                self.end_conv_1 = nn.Conv2d(in_channels=skip_channels + self.prototype_shape[0],
                                            out_channels=end_channels,
                                            kernel_size=(1, 1),
                                            bias=False)
                if self.addaptive_coef:
                    self.coff_map_layer = nn.Linear(64 + self.prototype_per_class*self.output_dim, 1,
                                bias=True)
            else:
                self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                            out_channels=end_channels,
                                            kernel_size=(1, 1),
                                            bias=True)

        else:
            self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                        out_channels=end_channels,
                                        kernel_size=(1, 1),
                                        bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
        # TODO: Fix device
        if self.is_xai["xai"]:
            self.prototype_class_identity = torch.zeros(self.prototype_per_class * self.output_dim,
                                                        self.output_dim).to(device)
            if self.is_xai["proto"]:
                self.pre_last_layer = nn.Linear(64 + self.prototype_per_class * self.output_dim, 2,
                                                bias=False)
            else:
                self.pre_last_layer = nn.Linear(64, 2, bias=False)
            for j in range(self.output_dim * self.prototype_per_class):
                self.prototype_class_identity[j, j // self.prototype_per_class] = 1
            self.set_last_layer_incorrect_connection(-0.5)
            # initialize the last layer
            self.top_k = top_k
        self.idx = torch.arange(self.num_nodes).to(device)
        self.device = device

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.pre_last_layer.weight[:, : self.prototype_per_class * self.output_dim].data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)
        # # Experiment fixed weights
        # for param in self.pre_last_layer.parameters():
        #     param.requires_grad = False

    def prototype_distances(self, x):
        xp = torch.mm(x, torch.t(self.prototype_vectors)) # x shape torch.Size([B, D])
        distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(self.prototype_vectors ** 2, dim=1, keepdim=True))
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        return similarity, distance

    def gumbel_softmax(self, logits):
        return F.gumbel_softmax(logits, tau=1, dim=-1)

    def gcn_encode(self, input, idx):
        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field - self.seq_length, 0, 0, 0))

        # if self.gcn_true:
        if self.buildA_true:
            if idx is None:
                adp: torch.Tensor = self.gc(self.idx)
            else:
                adp: torch.Tensor = self.gc(idx)
        else:
            adp: torch.Tensor = self.predefined_A(idx)  # TODO: (idx) --> for sub_grapf augmentaion

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            # if self.gcn_true:
            hiddens = []
            BATCH, _, _, TIME = x.shape
            if not self.gcn_true:
                for k in range(BATCH):
                    for w in range(TIME):
                        temp = self.gconv1[i](x[k, ..., w].t(), dense_to_sparse(adp)[0]) + self.gconv2[i](x[k, ..., w].t(),
                        dense_to_sparse(
                            adp.transpose(
                                1, 0))[0])
                        hiddens.append(temp[0].unsqueeze(0).unsqueeze(3).contiguous())
                hiddens = torch.cat(hiddens, dim=0)
                x = hiddens.view(BATCH, -1, self.num_nodes, TIME) # torch.Size([64, 128, 207, 13])
            else:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0)) # torch.Size([64, 32, 207, 13])
            # else:
            # x = self.residual_convs[i](x)

            x = x[:, :32, ...] + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x_clone = x.clone()

        return x, adp, x_clone

    def g_sub(self, x):
        # gcn_encode

        ### Making G-sub
        node_feature = x.squeeze(-1).transpose(1,2)  # torch.Size([64, 64, 207, 1]) 207: Node, 64: Hidden, 64: Batch
        #this part is used to generate assignment matrix
        abstract_features_1 = torch.tanh(self.fully_connected_1(node_feature)) 
        assignment = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=-1)
        top_probablities, top_indicies = torch.topk(assignment[:, :, 0], self.top_k, dim=1)
        gumbel_assignment = self.gumbel_softmax(assignment)  # torch.Size([375, 2])
        BATCH, num_nodes, _ = gumbel_assignment.shape
        # noisy_graph_representation
        lambda_pos = gumbel_assignment[:, :, 0].unsqueeze(dim=2)
        lambda_neg = gumbel_assignment[:, :, 1].unsqueeze(dim=2)

        # This is the graph embedding
        active = lambda_pos > 0.5
        active = active.squeeze()
        active_node_index = active.nonzero().tolist()
        # KL_Loss

        static_node_feature = node_feature.clone().detach()  # torch.Size([64, 207, 64])
        node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=1)
        node_feature_mean = node_feature_mean.unsqueeze(1).repeat(1, num_nodes, 1)

        noisy_node_feature_mean = lambda_pos * node_feature + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std.unsqueeze(1).repeat(1, num_nodes, 1)

        noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
        # this part is used to add noise
        # node_feature = node_feature
        # node_emb = node_feature

        return (noisy_node_feature, noisy_node_feature_std, node_feature_std,
                num_nodes, noisy_node_feature_mean, node_feature_mean, assignment,
                BATCH, node_feature, gumbel_assignment, top_indicies, active, active_node_index)

    def forward(self, input, idx=None, k_sparsity=0.4, ARGS=None):
        """
        input shape: B * d * N * T
        """

        x, adp, x_clone = self.gcn_encode(input, idx)

        #########################################################
        # G_sub & KL_loss
        if self.is_xai["xai"]:
            if self.is_xai["gsub"]:
                (noisy_node_feature, noisy_node_feature_std, node_feature_std,
                num_nodes, noisy_node_feature_mean, node_feature_mean, assignment,
                BATCH, node_feature, gumbel_assignment, top_indicies, active, active_node_index) = self.g_sub(x)
            else:
                (noisy_node_feature, noisy_node_feature_std, node_feature_std,
            num_nodes, noisy_node_feature_mean, node_feature_mean, assignment,
            BATCH, node_feature, gumbel_assignment, top_indicies, active, active_node_index) = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x, 0.0, 0.0, 0.0, 0.0
        else:
            (noisy_node_feature, noisy_node_feature_std, node_feature_std,
            num_nodes, noisy_node_feature_mean, node_feature_mean, assignment,
            BATCH, node_feature, gumbel_assignment, top_indicies, active, active_node_index) = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x, 0.0, 0.0, 0.0, 0.0
        epsilon = 0.0000001
        try:
            KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (
                    node_feature_std.unsqueeze(1).repeat(1, num_nodes, 1) + epsilon) ** 2) + \
                        torch.sum(((noisy_node_feature_mean - node_feature_mean) / (
                                node_feature_std.unsqueeze(1).repeat(1, num_nodes, 1) + epsilon)) ** 2, dim=0)
            KL_Loss = torch.mean(KL_tensor)  # Log(A) + log(B)
        except AttributeError:
            KL_Loss = torch.tensor(0.0, device=node_feature.device)
        #######################################################
        # connectivity loss
        #TODO: self.predefined_A --> for connectivity_loss (important)
        Adj = adp.clone().detach()
        Adj.requires_grad = False

        try:
            (assignment.transpose(1, 2) @ Adj)
        except AttributeError:
            pass
        except Exception as e:
            import pdb
            pdb.set_trace()
        try:
            new_adj = (assignment.transpose(1, 2) @ Adj)
            new_adj = new_adj @ assignment
            normalize_new_adj = F.normalize(new_adj, p=1, dim=1)
            norm_diag = torch.stack(
                [torch.diag(torch.diagonal(normalize_new_adj[i])) for i in range(normalize_new_adj.shape[0])], dim=0)

            if torch.cuda.is_available():
                # TODO: Fix device
                EYE = torch.eye(2).to(self.device).repeat(BATCH, 1, 1)
            else:
                EYE = torch.eye(2).repeat(BATCH, 1, 1)

            pos_penalty = torch.nn.MSELoss()(norm_diag, EYE)  # Eq. 15 TODO EYES ?
        except AttributeError:
            pos_penalty = torch.tensor(0.0, device=node_feature.device)
        #########################################################
        # congestion
        try:
            if self.mean_pool:
                graph_emb = noisy_node_feature.mean(dim=1)
            else:
                graph_emb, _ = noisy_node_feature.max(dim=1)
            if self.graph_reg:
                receivers = torch.matmul(self.rel_rec.to(noisy_node_feature.device), noisy_node_feature)
                senders = torch.matmul(self.rel_send.to(noisy_node_feature.device), noisy_node_feature)
                edge_feat = torch.cat([senders, receivers], dim=-1)
                edge_feat = torch.relu(self.fc_out(edge_feat))
                # Bernoulli parameter (unnormalized) Theta_{ij} in Eq. (2)
                bernoulli_unnorm = self.fc_cat(edge_feat)
                sampled_adj = F.gumbel_softmax(bernoulli_unnorm, tau=0.5, hard=True)[...,0].reshape(noisy_node_feature.shape[0], num_nodes, -1)
                mask = torch.eye(num_nodes, num_nodes).unsqueeze(0).bool().to(sampled_adj.device)
                sampled_adj.masked_fill_(mask, 0)
            prototype_activations, min_distance = self.prototype_distances(graph_emb)
        except AttributeError:
            graph_emb = node_feature
            try:
                prototype_activations, min_distance = self.prototype_distances(graph_emb.max(dim=2)[0].squeeze(-1))
            except AttributeError:
                prototype_activations, min_distance = 0.0, 0.0
        if self.is_xai["xai"]:
            if self.is_xai["proto"]:
                try:
                    final_embedding = torch.cat(
                        (prototype_activations.unsqueeze(1).repeat(1, node_feature.shape[1], 1), node_feature), dim=2)
                except RuntimeError:
                    final_embedding = torch.cat(
                    (prototype_activations.unsqueeze(1).repeat(1, node_feature.shape[1], 1), node_feature.squeeze(-1)), dim=2)
            else:
                final_embedding = node_feature
        else:
            final_embedding = node_feature
        try:
            congestion = F.softmax(self.pre_last_layer(final_embedding), dim=-1)
        except (AttributeError, RuntimeError):
            congestion = torch.tensor(0.0, device=node_feature.device)

        #########################################################
        # for Regression head
        if self.is_xai["xai"]:
            adaptive_coff = 1.0
            if self.addaptive_coef:
                adaptive_coff = (F.selu(self.coff_map_layer(final_embedding)) + 2.0).unsqueeze(1)
            x = F.relu(self.end_conv_1(final_embedding.transpose(1, 2).unsqueeze(-1)))
            x = self.end_conv_2(x) * adaptive_coff # Be positive
            if self.mi_loss:
                x = F.relu(x)
        else:
            x = F.relu(self.end_conv_1(final_embedding))
            x = self.end_conv_2(x)

        ##########################################################
        # prototype_pred_loss
        if self.is_xai["xai"]:
            for i in range(graph_emb.shape[0]):  # Eq. 10
                predicted_prototype = self.prototype_predictor(torch.t(graph_emb[i])).reshape(-1,
                                                                                            self.prototype_vectors.shape[
                                                                                                1])
                if i == 0:
                    prototype_pred_losses = self.loss_fn(self.prototype_vectors, predicted_prototype).reshape(1)
                else:
                    prototype_pred_losses = torch.cat(
                        (prototype_pred_losses, self.loss_fn(self.prototype_vectors, predicted_prototype).reshape(1)))
            prototype_pred_loss = prototype_pred_losses.mean()
        else:
            prototype_pred_loss = torch.tensor(0.0, device=node_feature.device)
        ###########################################################
        # Top-k
        # Complementarity prediction
        if self.is_xai["xai"] and self.is_xai["gsub"] & self.is_xai["proto"]:

            node_feature_c = x_clone.squeeze(-1).transpose(1,2)  # torch.Size([64, 64, 207, 1]) 207: Node, 64: Hidden, 64: Batch
            # this part is used to generate assignment matrix
            abstract_features_1_c = torch.tanh(self.fully_connected_1(node_feature_c))

            assignment_c = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1_c), dim=-1)
            BATCH, num_nodes, _ = assignment_c.shape


            #############################################

            static_node_feature = node_feature_c  # torch.Size([64, 207, 64])
            node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=1)
            node_feature_mean = node_feature_mean.unsqueeze(1).repeat(1, num_nodes, 1)

            #############################################
            Sub_nodes_prob = assignment_c[:, :, 0]
            Supp_nodes_prob = assignment_c[:, :, 1]

            top_prob_pose, top_ind_pose = torch.topk(Sub_nodes_prob, math.floor(k_sparsity * num_nodes), dim=1)
            # top_prob_neg, top_ind_neg = torch.topk(Supp_nodes_prob, num_nodes - math.floor(k_sparsity * num_nodes), dim=1)

            Sub_lambda = torch.zeros_like(Sub_nodes_prob)
            ones = torch.ones_like(top_ind_pose, dtype=Sub_nodes_prob.dtype)
            Sub_lambda.scatter_(dim=1, index=top_ind_pose, src=ones) #TODO: Check this
            Sub_1_lambda = 1.0 - Sub_lambda #TODO: Check this


            # Supp_lambda = torch.zeros_like(Supp_nodes_prob)
            # ones = torch.ones_like(top_ind_neg, dtype=Supp_nodes_prob.dtype)
            # Supp_lambda.scatter_(dim=1, index=top_ind_neg, src=ones)
            # Supp_1_lambda = 1.0 - Supp_lambda


            Supp_lambda = Sub_1_lambda
            Supp_1_lambda = Sub_lambda




            all_lambda = torch.ones_like(Sub_nodes_prob)
            all_1_lambda = torch.zeros_like(Sub_nodes_prob)

            #############################################
            assert (Sub_lambda + Supp_lambda == torch.ones_like(Supp_lambda).to(Sub_lambda.device)).all().item()
            comp_preds = []
            for lambda_pos, lambda_neg in zip([Sub_lambda, Supp_lambda, all_lambda], [Sub_1_lambda, Supp_1_lambda, all_1_lambda]):

                y = self.topk_out(lambda_pos.unsqueeze(dim=-1), lambda_neg.unsqueeze(dim=-1), node_feature_c, node_feature_mean, node_feature_std, num_nodes)

                comp_preds.append(y)

            ##############################################
            # Sparsity

            # noisy_graph_representation
            lambda_pos = gumbel_assignment[:, :, 0].unsqueeze(dim=2)
            # lambda_neg = gumbel_assignment[:, :, 1].unsqueeze(dim=2)
            # This is the graph embedding
            active = lambda_pos > 0.5
            active = active.squeeze()
            active_node_index = active.nonzero().tolist()
            # KL_Loss
            sparsity = (1 - torch.sum(active, dim=1) / x_clone.size(2)).mean()
            ##############################################

        else:
            comp_preds = [0.0, 0.0, 0.0]
            sparsity = 0.0


    ##############################################################################################
        if locals().get('sampled_adj') is not None:
            sampled_adj = locals().get('sampled_adj')
        else:
            sampled_adj = None
        return (x, active_node_index, graph_emb, KL_Loss, pos_penalty, prototype_pred_loss,
                min_distance, congestion, adp, gumbel_assignment, comp_preds, sparsity, sampled_adj)

    # def populate_by_topk_lambdas(self, probs: torch.Tensor, indicies: torch.Tensor,
    #                              device: torch.cuda.device) -> torch.Tensor:
    #     """
    #
    #     """
    #     # lambdas = torch.zeros_like(probs)
    #     # mask = torch.zeros_like(lambdas, dtype=torch.bool)
    #     # row_indices = torch.arange(lambdas.shape[0]).unsqueeze(1).expand(-1, indicies.shape[1])
    #     # mask[row_indices, indicies, 0] = True
    #     # lambdas[mask] = torch.tensor([1.0, 0.0], device=device)
    #     # lambdas[~mask] = torch.tensor([0.0, 1.0], device=device)
    #     # lambda_pos = lambdas[...,0]
    #     # lambda_neg = lambdas[...,1]
    #     # return lambda_pos, lambda_neg
    #     gumbel_assignment = self.gumbel_softmax(probs)  # torch.Size([64, 207, 2])
    #     # noisy_graph_representation
    #     lambda_pos = gumbel_assignment[:, :, 0].unsqueeze(dim=2)  # torch.Size([64, 207, 1])
    #     lambda_neg = gumbel_assignment[:, :, 1].unsqueeze(dim=2)  # torch.Size([64, 207, 1])
    #     return lambda_pos, lambda_neg
    #
    # def populate_by_gumbel_lambdas(self, assigns: torch.Tensor) -> torch.Tensor:
    #     """
    #
    #     """
    #     gumbel_assignment = self.gumbel_softmax(assigns)  # torch.Size([64, 207, 2])
    #     # noisy_graph_representation
    #     lambda_pos = gumbel_assignment[:, :, 0].unsqueeze(dim=2)  # torch.Size([64, 207, 1])
    #     lambda_neg = gumbel_assignment[:, :, 1].unsqueeze(dim=2)  # torch.Size([64, 207, 1])
    #     return lambda_pos, lambda_neg

    def topk_out(self, lambda_pos, lambda_neg, node_feature, node_feature_mean, node_feature_std, num_nodes):
        noisy_node_feature_mean = lambda_pos * node_feature + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std.unsqueeze(1).repeat(1, num_nodes, 1)

        noisy_node_feature = noisy_node_feature_mean + torch.rand_like(
            noisy_node_feature_mean) * noisy_node_feature_std

        ###################################################

        try:
            if self.mean_pool:
                graph_emb = noisy_node_feature.mean(dim=1)
            else:
                graph_emb, _ = noisy_node_feature.max(dim=1)
            if self.graph_reg:
                receivers = torch.matmul(self.rel_rec.to(noisy_node_feature.device), noisy_node_feature)
                senders = torch.matmul(self.rel_send.to(noisy_node_feature.device), noisy_node_feature)
                edge_feat = torch.cat([senders, receivers], dim=-1)
                edge_feat = torch.relu(self.fc_out(edge_feat))
                # Bernoulli parameter (unnormalized) Theta_{ij} in Eq. (2)
                bernoulli_unnorm = self.fc_cat(edge_feat)
                sampled_adj = F.gumbel_softmax(bernoulli_unnorm, tau=0.5, hard=True)[..., 0].reshape(
                    noisy_node_feature.shape[0], num_nodes, -1)
                mask = torch.eye(num_nodes, num_nodes).unsqueeze(0).bool().to(sampled_adj.device)
                sampled_adj.masked_fill_(mask, 0)
            prototype_activations, min_distance = self.prototype_distances(graph_emb)
        except AttributeError:
            graph_emb = node_feature
            try:
                prototype_activations, min_distance = self.prototype_distances(
                    graph_emb.max(dim=2)[0].squeeze(-1))
            except AttributeError:
                prototype_activations, min_distance = 0.0, 0.0
        if self.is_xai["xai"]:
            if self.is_xai["proto"]:
                try:
                    final_embedding_c = torch.cat(
                        (prototype_activations.unsqueeze(1).repeat(1, node_feature.shape[1], 1), node_feature),
                        dim=2)
                except RuntimeError:
                    final_embedding_c = torch.cat(
                        (prototype_activations.unsqueeze(1).repeat(1, node_feature.shape[1], 1),
                         node_feature.squeeze(-1)), dim=2)
            else:
                final_embedding_c = node_feature
        else:
            final_embedding_c = node_feature

        ###################################################

        if self.is_xai["xai"]:
            adaptive_coff_c = 1.0
            if self.addaptive_coef:
                adaptive_coff_c = (F.selu(self.coff_map_layer(final_embedding_c)) + 2.0).unsqueeze(1)
            x = F.relu(self.end_conv_1(final_embedding_c.transpose(1, 2).unsqueeze(-1)))
            x = self.end_conv_2(x) * adaptive_coff_c  # Be positive
            if self.mi_loss:
                x = F.relu(x)
        else:
            x = F.relu(self.end_conv_1(final_embedding_c))
            x = self.end_conv_2(x)

        return x
