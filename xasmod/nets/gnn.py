import sys
import torch
import torch_geometric
import torch.nn.functional as F
from torch_scatter import scatter
from scipy.interpolate import interp1d
from torch.nn import Sequential, Linear, BatchNorm1d
from torch_geometric.nn import MessagePassing 
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import (Set2Set,global_add_pool,global_max_pool,GCNConv)


class Three_Sections_GNN(torch.nn.Module):
    def __init__(
        self,
        data,
        dims=[[200,200,200],[200],[100]], # GCN,MHGA,MLP(n-1)
        heads = 3,
        batch_norm=True,
        batch_track_stats="True",
        dropout_rate=0.0,
        edge_features = 1,
        **kwargs
    ):
        super(Three_Sections_GNN, self).__init__()
        self.batch_track_stats = batch_track_stats 
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.input_dim = data.num_features
        self.trans_dim = data.num_features
        self.output_dim = len(data[0].y)
        add_self_loop = True
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        for i in dims[0]:
            conv = SimpleGraphConv(
                            in_channels=self.trans_dim, 
                            out_channels=i, 
                            edge_features=edge_features, 
                            add_self_loops=add_self_loop
                            )
            self.conv_list.append(conv)
            if self.batch_norm:
                bn = BatchNorm1d(i, track_running_stats=self.batch_track_stats, affine=True)
                self.bn_list.append(bn)
            self.trans_dim = i
        self.heads = heads
        self.att_list = torch.nn.ModuleList()
        self.bn2_list = torch.nn.ModuleList()
        for i in dims[1]:
            mattlist = torch.nn.ModuleList()
            mbn2list = torch.nn.ModuleList()
            for j in range(0,self.heads):
                att = OneHeadGraphAttetion(
                                in_channels=self.trans_dim, 
                                out_channels=i, 
                                edge_features=edge_features, 
                                add_self_loops=add_self_loop
                                )
                mattlist.append(att)
                if self.batch_norm:
                    bn = BatchNorm1d(i, track_running_stats=self.batch_track_stats, affine=True)
                    mbn2list.append(bn)
                

            self.att_list.append(mattlist)
            self.bn2_list.append(mbn2list)
            self.trans_dim = i
        if len(dims[1]) !=0:
            self.trans_dim = self.trans_dim * self.heads
        self.post_mlp_list = torch.nn.ModuleList()
        for i in dims[2]:
            lin = Sequential(torch.nn.Linear(self.trans_dim, i), torch.nn.PReLU())
            self.post_mlp_list.append(lin)
            self.trans_dim = i
        lin = Sequential(torch.nn.Linear(self.trans_dim, self.output_dim))
        self.post_mlp_list.append(lin)


    def forward(self, data):
        
        

        out = data.x
        for i in range(0, len(self.conv_list)):
            out = self.conv_list[i](out, data.edge_index, data.edge_attr)
            if self.batch_norm:
                out = self.bn_list[i](out)
        first_mhalayer = True
        for i in range(0, len(self.att_list)):
            if first_mhalayer == True:
                o = [out]*self.heads
                first_mhalayer = False
            o=[self.att_list[i][jid](j,data.edge_index,data.edge_attr) for jid,j in enumerate(o)] 
            if self.batch_norm:
                o = [self.bn2_list[i][jid](j) for jid,j in enumerate(o)]
        if len(self.att_list) != 0:
            out = torch.cat(o, dim=1)
        out = global_mean_pool(out, data.batch)
        out = F.dropout(out, p=self.dropout_rate, training=self.training)
        for i in range(0, len(self.post_mlp_list)):
            out = self.post_mlp_list[i](out)
        out = out.view(-1)

        return out
    

class SimpleGraphConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_features,add_self_loops=False, bias=False):
        super(SimpleGraphConv, self).__init__()
        self.add_self_loops = add_self_loops
        self.edgelinear = Sequential(torch.nn.Linear(edge_features, in_channels, bias=True),torch.nn.ELU())
        self.updatalinear = Sequential(torch.nn.Linear(in_channels, out_channels, bias=True))
        return


    def message(self, x, edge_index, edge_attr):
        mask = edge_attr[:, 0].unsqueeze(1).contiguous() < 8
        edge_sources, edge_targets = edge_index
        edge_attr_vector = edge_attr[:, :65]
        edge = self.edgelinear(edge_attr_vector)
        m = edge * x[edge_targets].contiguous() * mask
        return m
    

    def aggregate(self, m, edge_index):
        edge_sources, _ = edge_index
        aggr_out = scatter(m, edge_sources, dim=0, reduce='sum')
        return aggr_out
    

    def update(self, x, aggr_out):
        x = x + aggr_out
        x = self.updatalinear(x)
        return x
    

    def propagate(self, x, edge_index, edge_attr):
        if self.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                                                    edge_index, 
                                                    edge_attr, 
                                                    num_nodes=x.size(0), 
                                                    fill_value=0
                                                    )
        m = self.message(x=x, edge_index=edge_index, edge_attr=edge_attr)
        aggr_out = self.aggregate(m, edge_index)
        output = self.update(x, aggr_out)
        return output
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)


class OneHeadGraphAttetion(torch.nn.Module):

    def __init__(self, in_channels, out_channels, edge_features, add_self_loops=False, bias=False):
        super(OneHeadGraphAttetion, self).__init__()
        self.add_self_loops = add_self_loops
        fcat_channels = in_channels * 3 + edge_features
        self.f_linear = Sequential(torch.nn.Linear(fcat_channels, in_channels, bias=True), torch.nn.ELU())
        qcat_channels = in_channels 
        kcat_channels = in_channels
        self.qk_channels = in_channels
        self.qlinear=torch.nn.Linear(qcat_channels, self.qk_channels, bias=False)
        self.klinear=torch.nn.Linear(kcat_channels, self.qk_channels, bias=False)
        self.alinear=Sequential(torch.nn.Linear(self.qk_channels, 1, bias=False), torch.nn.Tanh())
        self.updatalinear = Sequential(torch.nn.Linear(in_channels, out_channels, bias=True))
        return

    def message(self, x, edge_index, edge_attr):
        edge_sources, edge_targets = edge_index
        mask = edge_attr[:, 0].unsqueeze(1).contiguous() < 8
        edge_attr_vector = edge_attr[:, :65]
        ni = x[edge_sources].contiguous()
        nj = x[edge_targets].contiguous()
        delta = ni-nj
        fcat_vector = torch.cat([ni,nj,delta,edge_attr_vector], dim=1)
        del ni,nj,delta
        torch.cuda.empty_cache()
        f = self.f_linear(fcat_vector) * mask
        qk = self.qk_channels ** (-0.5)
        q = self.qlinear(x[edge_sources].contiguous())
        k = self.klinear(f)
        a = self.alinear(q * k * qk) 
        numerator = torch.exp(a)
        denominator = torch.exp(scatter(a, edge_sources, dim=0, reduce='sum')[edge_sources].contiguous())
        s = numerator / denominator
        z = s * f
        return z
    

    def aggregate(self, z, edge_index):
        edge_sources, _ = edge_index
        aggr_out = scatter(z, edge_sources, dim=0, reduce='sum')
        return aggr_out
    

    def update(self, x, aggr_out):
        x = x + aggr_out
        x = self.updatalinear(x)
        return x
    

    def propagate(self, x, edge_index, edge_attr):
        if self.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                                                    edge_index, 
                                                    edge_attr, 
                                                    num_nodes=x.size(0), 
                                                    fill_value=0
                                                    )
        z = self.message(x=x, edge_index=edge_index, edge_attr=edge_attr)
        aggr_out = self.aggregate(z, edge_index)
        output = self.update(x, aggr_out)
        return output


    def forward(self, x, edge_index, edge_attr):
        return self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
