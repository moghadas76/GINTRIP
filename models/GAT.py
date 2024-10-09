import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import to_dense_adj
from Configures import data_args, train_args, model_args
import pdb
from itertools import accumulate
from typing import Union, Optional, List, Dict
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, PairTensor

import torch
from torch import Tensor
from torch_geometric.nn import GINEConv as BaseGINEConv, GINConv as BaseGINConv, LEConv as BaseLEConv, GATConv as BaseGATConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import degree
from torch_scatter import scatter
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch.nn.functional as F
from torch_sparse import SparseTensor, set_diag


def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


# GAT
class GATNet(nn.Module):
    def __init__(self, input_dim, output_dim, model_args):
        super(GATNet, self).__init__()
        self.output_dim = output_dim
        self.latent_dim = model_args.latent_dim
        self.device = model_args.device
        self.num_gnn_layers = model_args.num_gat_layer
        self.dense_dim = model_args.gat_hidden * model_args.gat_heads
        self.readout_layers = get_readout_layers(model_args.readout)
        # self.tau1 = model_args.tau1   
        # self.tau2 = 1
        
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GATConv(input_dim, model_args.gat_hidden, heads=model_args.gat_heads,
                                       dropout=model_args.gat_dropout, concat=model_args.gat_concate))
        
        for i in range(1, self.num_gnn_layers):
            self.gnn_layers.append(GATConv(self.dense_dim, model_args.gat_hidden, heads=model_args.gat_heads,
                                           dropout=model_args.gat_dropout, concat=model_args.gat_concate))

        
        self.fully_connected_1 = torch.nn.Linear(self.dense_dim, self.dense_dim)
        self.fully_connected_2 = torch.nn.Linear(self.dense_dim, 2)

        
        self.Softmax = nn.Softmax(dim=-1)

        # prototype layers
        self.enable_prot = model_args.enable_prot
        self.epsilon = 1e-4
        self.prototype_shape = (output_dim * model_args.num_prototypes_per_class, self.dense_dim)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        self.num_prototypes = self.prototype_shape[0]
        self.prototype_predictor = nn.Linear(self.dense_dim, self.num_prototypes*self.dense_dim, bias=False)
        self.mse_loss = torch.nn.MSELoss()
        
        self.last_layer = nn.Linear(self.latent_dim[2] + self.num_prototypes, output_dim,
                                    bias=False)  # do not use bias
        assert (self.num_prototypes % output_dim == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    output_dim)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // model_args.num_prototypes_per_class] = 1
        # initialize the last layer
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight[:,: self.num_prototypes].data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def prototype_distances(self, x):
        xp = torch.mm(x, torch.t(self.prototype_vectors))
        distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(self.prototype_vectors ** 2, dim=1, keepdim=True)) 
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        return similarity, distance
  

    def forward(self, data, merge=False):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch 

        node_features_1 = torch.nn.functional.relu(self.gnn_layers[0](x, edge_index)) 
        node_features_2 = self.gnn_layers[1](node_features_1, edge_index) 
        num_nodes = node_features_2.size()[0] 

        #this part is used to add noise
        node_feature = node_features_2 
        node_emb = node_feature

        #this part is used to generate assignment matrix
        abstract_features_1 = torch.tanh(self.fully_connected_1(node_feature)) 
        assignment = torch.nn.functional.softmax(self.fully_connected_2(abstract_features_1), dim=1) 


        gumbel_assignment = self.gumbel_softmax(assignment) 

        #noisy_graph_representation
        lambda_pos = gumbel_assignment[:,0].unsqueeze(dim = 1) 
        lambda_neg = gumbel_assignment[:,1].unsqueeze(dim = 1) 

        
        #This is the graph embedding
        active = lambda_pos>0.5
        active = active.squeeze()

        
        active_node_index=[]
        node_number = [0]
        for i in range(batch[-1]+1): 
            node_number.append(len(batch[batch==i]))
        
        node_number = list(accumulate(node_number))
        for j in range(len(node_number)-1):
            active_node_index.append(active[node_number[j]:node_number[j+1]].nonzero().squeeze().tolist()) 

        #KL_Loss 
        static_node_feature = node_feature.clone().detach() 
        node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0) 
        node_feature_mean = node_feature_mean.repeat(num_nodes,1) 

        noisy_node_feature_mean = lambda_pos * node_feature + lambda_neg * node_feature_mean 
        noisy_node_feature_std = lambda_neg * node_feature_std 

        noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std #

        for readout in self.readout_layers:
            noisy_graph_feature = readout(noisy_node_feature, batch)
        
        graph_emb = noisy_graph_feature

        epsilon = 0.0000001
        KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std+epsilon) ** 2) + \
                    torch.sum(((noisy_node_feature_mean -node_feature_mean)/(node_feature_std + epsilon))**2, dim = 0) 
        KL_Loss = torch.mean(KL_tensor)  

        Adj = to_dense_adj(edge_index, max_num_nodes=assignment.shape[0])[0]
        Adj.requires_grad = False
        new_adj = torch.mm(torch.t(assignment),Adj)
        new_adj = torch.mm(new_adj,assignment)
        normalize_new_adj = F.normalize(new_adj, p=1, dim=1)
        norm_diag = torch.diag(normalize_new_adj)

        if torch.cuda.is_available():
            EYE = torch.ones(2).cuda()
        else:
            EYE = torch.ones(2)

        pos_penalty = torch.nn.MSELoss()(norm_diag, EYE)

        ## graph embedding
        prototype_activations, min_distance = self.prototype_distances(graph_emb) 

        final_embedding = torch.cat((prototype_activations, graph_emb), dim=1)###########################################################
        logits = self.last_layer(final_embedding)
        probs = self.Softmax(logits)

        if model_args.cont:
            return logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, prototype_activations, min_distance 
        
        else:
            for i in range(graph_emb.shape[0]):
                predicted_prototype = self.prototype_predictor(torch.t(graph_emb[i])).reshape(-1, self.prototype_vectors.shape[1]) 
                if i == 0:
                    prototype_pred_losses = self.mse_loss(self.prototype_vectors, predicted_prototype).reshape(1)
                else:
                    prototype_pred_losses = torch.cat((prototype_pred_losses,self.mse_loss(self.prototype_vectors, predicted_prototype).reshape(1)))
            prototype_pred_loss = prototype_pred_losses.mean()

            return logits, probs, active_node_index, graph_emb, KL_Loss, pos_penalty, prototype_pred_loss, min_distance 

    
    def gumbel_softmax(self, prob):

        return F.gumbel_softmax(prob, tau = 1, dim = -1)




class GATConvExp(BaseGATConv):
    """
    This GAT came from STExplainer paper
    """
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None, edge_atten: OptTensor = None):
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        # print(x.shape)
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)
        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor, edge_attr: OptTensor)  # noqa
        # print(edge_index.dtype, edge_index.shape)
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr,
                             size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias
        
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            # return out
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int], edge_atten: OptTensor = None) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        if edge_atten is None:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        else:
            alpha = alpha * edge_atten
        return x_j * alpha.unsqueeze(-1)





if __name__ == "__main__":
    from Configures import model_args
    model = GATNet(7, 2, model_args)
    pass