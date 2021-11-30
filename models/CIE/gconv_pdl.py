import math
import time
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Gconv(nn.Layer):
    """
    (Intra) graph convolution operation, with single convolutional layer
    """
    def __init__(self, in_features, out_features):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        k = math.sqrt(1.0 / in_features)
        weight_attr_1 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_1 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        weight_attr_2 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_2 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))

        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs,
                                weight_attr=weight_attr_1, 
                                bias_attr=bias_attr_1)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs,
                                weight_attr=weight_attr_2, 
                                bias_attr=bias_attr_2)

    def forward(self, A, x, norm=True):
        if norm is True:
            A = F.normalize(A, p=1, axis=-2)

        ax = self.a_fc(x)
        ux = self.u_fc(x)
        x = paddle.bmm(A, F.relu(ax)) + F.relu(ux) # has size (bs, N, num_outputs)
        return x

class Siamese_Gconv(nn.Layer):
    """
    Perform graph convolution on two input graphs (g1, g2)
    """
    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1, g2):
        emb1 = self.gconv(*g1)
        emb2 = self.gconv(*g2)
        # embx are tensors of size (bs, N, num_features)
        return emb1, emb2

class ChannelIndependentConv(nn.Layer):

    def __init__(self, in_features, out_features, in_edges, out_edges=None):
        super(ChannelIndependentConv, self).__init__()
        if out_edges is None:
            out_edges = out_features
        self.in_features = in_features
        self.out_features = out_features
        self.out_edges = out_edges
        k = math.sqrt(1.0 / in_features)
        
        weight_attr_1 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_1 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        weight_attr_2 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_2 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        weight_attr_3 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        bias_attr_3 = paddle.ParamAttr(
                initializer=paddle.nn.initializer.Uniform(-k, k))
        
        self.node_fc = nn.Linear(in_features, out_features, weight_attr=weight_attr_1, bias_attr=bias_attr_1)
        self.node_sfc = nn.Linear(in_features, out_features, weight_attr=weight_attr_2, bias_attr=bias_attr_2)
        self.edge_fc = nn.Linear(in_edges, self.out_edges, weight_attr=weight_attr_3, bias_attr=bias_attr_3)

    def forward(self, A, emb_node, emb_edge, mode=1):

        if mode == 1:
            node_x = self.node_fc(emb_node)
            node_sx = self.node_sfc(emb_node)
            edge_x = self.edge_fc(emb_edge)

            A = A.unsqueeze(-1)
            A = paddle.multiply(A.expand_as(edge_x), edge_x)
            node_x = paddle.matmul(A.transpose((0,1,3,2)).transpose((0,2,1,3)),
                                  node_x.unsqueeze(2).transpose((0,1,3,2)).transpose((0,2,1,3)))
            node_x = node_x.squeeze(-1).transpose((0,2,1))
            node_x = F.relu(node_x) + F.relu(node_sx)
            edge_x = F.relu(edge_x)

            return node_x, edge_x

        elif mode == 2:
            node_x = self.node_fc(emb_node)
            node_sx = self.node_sfc(emb_node)
            edge_x = self.edge_fc(emb_edge)

            d_x = node_x.unsqueeze(1) - node_x.unsqueeze(2)
            d_x = paddle.sum(d_x ** 2, dim=3, keepdim=False)
            d_x = paddle.exp(-d_x)

            A = A.unsqueeze(-1)
            A = paddle.multiply(A.expand_as(edge_x), edge_x)

            node_x = paddle.matmul(A.transpose((0,1,3,2)).transpose((0,2,1,3)),
                                  node_x.unsqueeze(2).transpose((0,1,3,2)).transpose((0,2,1,3)))
            node_x = node_x.squeeze(-1).transpose((0,2,1))
            node_x = F.relu(node_x) + F.relu(node_sx)
            edge_x = F.relu(edge_x)
            return node_x, edge_x

class Siamese_ChannelIndependentConv(nn.Layer):

    def __init__(self, in_features, num_features, in_edges, out_edges=None):
        super(Siamese_ChannelIndependentConv, self).__init__()
        self.in_feature = in_features
        self.gconv = ChannelIndependentConv(in_features, num_features, in_edges, out_edges)

    def forward(self, g1, *args):
 
        emb1, emb_edge1 = self.gconv(*g1)
        embs = [emb1]
        emb_edges = [emb_edge1]
        for g in args:
            emb2, emb_edge2 = self.gconv(*g)
            embs.append(emb2), emb_edges.append(emb_edge2)
        return embs + emb_edges
