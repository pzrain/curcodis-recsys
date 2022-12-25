import argparse
import operator
import random

import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim

import numpy as np
import scipy.sparse as sparse
import networkx as nx
import bottleneck as bn
import os
import time
import matplotlib.pyplot as plt

from difficulty import calc_difficulty
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parseArgs():
    ARG = argparse.ArgumentParser()
    ARG.add_argument('--data', type=str, default='douban',
                     help='../dataset/lastfm, ../dataset/duoban, ../dataset/Ciao, ../dataset/Epinion, ../dataset/yelp', )
    ARG.add_argument('--epoch', type=int, default=160,
                     help='Number of maximum training epochs.')
    ARG.add_argument('--batchNum', type=int, default=2,
                 help='Training Number of batch.')
    ARG.add_argument('--lr', type=float, default=1e-3,
                     help='Initial learning rate.')
    ARG.add_argument('--rg', type=float, default=0.0,
                     help='L2 regularization.')
    ARG.add_argument('--keep', type=float, default=0.5,
                     help='Keep probability for dropout in VAE, in (0,1].')
    ARG.add_argument('--beta', type=float, default=0.2,
                     help='Strength of disentanglement, in (0,oo).')
    ARG.add_argument('--tau', type=float, default=0.1,
                     help='Temperature of sigmoid/softmax, in (0,oo).')
    ARG.add_argument('--std', type=float, default=0.075,
                     help='Standard deviation of the Gaussian prior.')
    ARG.add_argument('--kfac', type=int, default=7,
                     help='Number of facets (macro concepts).')
    ARG.add_argument('--dfac', type=int, default=250,
                     help='Dimension of each facet.')
    ARG.add_argument('--nogb', action='store_true', default=False,
                     help='Disable Gumbel-Softmax sampling.')
    ARG.add_argument('--numLayer', type=int, default=1,
                     help='Number of residual blocks.')
    ARG.add_argument("--numGCNLayer", type=int, default=5,
                     help="Number of disenGCN layers")
    ARG.add_argument('--dropout', type=float, default=0.35,
                            help='Dropout rate (1 - keep probability) in DisenConv.')
    ARG.add_argument('--routit', type=int, default=10,
                            help='Number of iterations when routing.')
    ARG.add_argument('--nbsz', type=int, default=20,
                            help='Size of the sampled neighborhood.')
    ARG.add_argument('--early', type=int, default=20,
                        help='Extra iterations before early-stopping.')
    ARG.add_argument('--cpu', action='store_true', default=False,
                        help='Insist on using CPU instead of CUDA.')
    ARG.add_argument('--monorate', action='store_true', default=True,
                        help='Transform the input rates into 0/1.')
    ARG.add_argument('--split', type=float, default=0,
                     help='Proportion of validation data in the training dataset.')
    ARG.add_argument('--ratio', type=float, default=8,
                     help='Ratio between residual and DisenGCN.')
    ARG.add_argument('--partitionK', type=int, default=4, help="partition the graph into 2^k subgraphs")
    ARG.add_argument('--gpudevice', type=int, default=0)
    ARG = ARG.parse_args()

    return ARG

#在DisenGraph中初始化表征的input层，实际上没用到
class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):  # *nn.Linear* does not accept sparse *x*.
        return torch.mm(x, self.weight) + self.bias

#在DisenGCN层中用于采样邻居
class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False):
        n = graph.number_of_nodes()
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        if include_self:
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)
            nb = nb_all[:, 1:]
        else:
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))
                nb[v] = torch.LongTensor(nb_v)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all
        nb_size = nb.size(1)
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size) ## 邻居数量多于指定值的，随机去掉一部分
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all ## ?

#DisenGCN层中的路由传播部分
class RoutingLayer(nn.Module):
    def __init__(self, cap_sz, out_caps, inp_caps=None):
        super(RoutingLayer, self).__init__()
        if inp_caps is not None:
            self.fc = nn.Linear(cap_sz * inp_caps, cap_sz * out_caps)
        self.d, self.k = cap_sz * out_caps, out_caps
        self._cache_zero_d = torch.zeros(1, self.d)
        self._cache_zero_k = torch.zeros(1, self.k)

    def forward(self, x, neighbors, max_iter):
        dev = x.device
        if self._cache_zero_d.device != dev:
            self._cache_zero_d = self._cache_zero_d.to(dev)
            self._cache_zero_k = self._cache_zero_k.to(dev)
        if hasattr(self, 'fc'):
            x = fn.relu(self.fc(x))
        n, m = x.size(0), neighbors.size(0) // x.size(0)
        d, k, delta_d = self.d, self.k, self.d // self.k
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = torch.cat([x, self._cache_zero_d], dim=0)
        z = z[neighbors].view(n, m, k, delta_d)
        u = x.view(n, k, delta_d)  # u = None
        for clus_iter in range(max_iter):
            if u is None:
                p = self._cache_zero_k.expand(n * m, k).view(n, m, k)
            else:
                p = torch.sum(z * u.view(n, 1, k, delta_d), dim=3)
            p = fn.softmax(p, dim=2)  # p = fn.softmax(p / tau, dim=2)
            u = torch.sum(z * p.view(n, m, k, 1), dim=1)
            u += x.view(n, k, delta_d)
            if clus_iter < max_iter - 1:
                u = fn.normalize(u, dim=2)
        return u.view(n, d)

# DisenGCN层
# 相对于源代码默认jump=False
class Capsule(nn.Module):
    def __init__(self, nlayer, ncaps, nhidden, dropout=0.35, routit=6):
        super(Capsule, self).__init__()
        conv_ls = []
        inp_caps, out_caps = None, ncaps
        #这个pca层实际上没用到
        self.pca = SparseInputLinear(nhidden * ncaps, nhidden * ncaps)
        for i in range(nlayer):
            conv = RoutingLayer(nhidden, out_caps, inp_caps)
            self.add_module('conv_%-d' % i, conv)
            conv_ls.append(conv)
            inp_caps, out_caps = out_caps, out_caps
        self.conv_ls = conv_ls
        self.dropout = dropout
        self.routit = routit

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def forward(self, x, nb): ## nb: neighbor
        nb = nb.view(-1)
        for conv in self.conv_ls:
            x = self._dropout(fn.relu(conv(x, nb, self.routit)))
        return x

# 计算NDCG@K
def ndcg_binary_at_k_batch(x_pred, heldout_batch, k=100):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = x_pred.shape[0]
    idx_topk_part = bn.argpartition(-x_pred, k, axis=1)
    topk_part = x_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (heldout_batch.toarray()[np.arange(batch_users)[:, np.newaxis],
                         idx_topk] * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    ndcg = dcg / idcg
    ndcg = ndcg[~np.isnan(ndcg)]
    return ndcg

# 计算RECALL@K
def recall_at_k_batch(x_pred, heldout_batch, k=100):
    batch_users = x_pred.shape[0]
    idx = bn.argpartition(-x_pred, k, axis=1)
    x_pred_binary = np.zeros_like(x_pred, dtype=bool)
    x_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    x_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(x_true_binary, x_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, x_true_binary.sum(axis=1))
    recall = recall[~np.isnan(recall)]
    return recall

# 以上代码全部来自于DisenGCN或者Disentangle-recsys
# 源代码可以在 https://jianxinma.github.io/ 找到
# 可能在将tensorflow迁移到pytorch以及调试训练时进行了一些修改，但基本是一样的

# 带参数ratio的残差网络（或者可以理解为attention network）
# x: Disentangle-recsys的输出特征
# block: DisenGCN层
class RecursiveResidual(nn.Module):
    def __init__(self, block, numLayers, ratio):
        super().__init__()
        self.block = block
        self.numLayers = numLayers
        self.ratio = ratio

    def forward(self, x, nb):
        o = x
        for i in range(self.numLayers):
            o = self.ratio * x + self.block(o, nb) ## 这里的self.ratio可以理解为attention
        return o

# 以disentangle-recsys源码为基础的VAE
class myVAE(nn.Module):
    def __init__(self, num_items, ARG):
        super(myVAE, self).__init__()
        # 初始化参数
        kfac, dfac = ARG.kfac, ARG.dfac
        self.lam = ARG.rg
        self.lr = ARG.lr
        self.std = ARG.std
        self.nogb = ARG.nogb
        self.tau = ARG.tau
        self.kfac = ARG.kfac
        self.n_items = num_items
        self.input_ph = None
        self.anneal_ph = None
        self.is_training_ph = None
        self.device = ARG.device
        # 初始化网络层
        self.dropout = nn.Dropout(p=(1-ARG.keep)) ## randomly zeroes some of the element of the input tensor with probabiity p using samples from a Bernouli distribution
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.q_dims = [num_items, dfac, dfac]
        self.residual = RecursiveResidual(Capsule(ARG.numGCNLayer, kfac, dfac, ARG.dropout, ARG.routit), ARG.numLayer, ARG.ratio) ## numLayers: 多次经过DisenGCN，学习邻居的邻居关系
        self.items = torch.zeros([num_items, dfac])
        self.cores = torch.zeros([kfac, dfac])
        nn.init.xavier_uniform_(self.items)
        nn.init.xavier_uniform_(self.cores)
        self.items = nn.Parameter(self.items) ## requires_grad = True
        self.cores = nn.Parameter(self.cores)
        self.linears_q = []
        self.capsules = []
        # The first fc layer of the encoder Q is the context embedding table.
        for i, (d_in, d_out) in enumerate(
                zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                d_out *= 2  # mu & var
            linear = nn.Linear(d_in, d_out)
            nn.init.xavier_uniform_(linear.weight)
            self.truncated_normal_(linear.bias,std=0.001)
            self.linears_q.append(linear)
            self.add_module('fc_%-d' % i, linear)

    #以下是两个用于初始化和归一化的工具函数
    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def l2_normalize(self, x, axis):
        norm = torch.norm(x, 2, axis, True)
        norm = norm + torch.full_like(norm, 1e-15) # 处理norm为0的情况
        x = torch.div(x, norm)
        return x

    # Encoder中预测特征所符合的正态分布的部分
    def q_graph_k(self, x):
        mu_q, std_q, kl = None, None, None
        h = self.l2_normalize(x, 1)
        h = self.dropout(h)
        l = len(self.linears_q)
        for i in range(l):
            linear = self.linears_q[i]
            h = linear(h)
            if i != l - 1:
                h = torch.tanh(h) ## 线性层中间夹的激活层
            else: ## 最后一层，计算得到结果
                mu_q = h[:, :self.q_dims[-1]]
                mu_q = self.l2_normalize(mu_q, axis=1)
                lnvarq_sub_lnvar0 = -h[:, self.q_dims[-1]:]
                std0 = self.std
                std_q = torch.exp(0.5 * lnvarq_sub_lnvar0) * std0
                # Trick: KL is constant w.r.t. to mu_q after we normalize mu_q.
                kl = torch.mean(torch.sum(
                    0.5 * (-lnvarq_sub_lnvar0 + torch.exp(lnvarq_sub_lnvar0) - 1.), 1))
        return mu_q, std_q, kl

    # 向前传播函数，大部分和disen-recsys是一样的
    def forward(self, save_emb, neighbors, input_ph=None, is_training_ph=None, anneal_ph=None):
        self.input_ph = input_ph
        self.is_training_ph = is_training_ph
        self.anneal_ph = anneal_ph

        ## ============
        # clustering
        cores = self.l2_normalize(self.cores, axis=1)
        items = self.l2_normalize(self.items, axis=1)
        cates_logits = torch.mm(items, torch.transpose(cores,0,1)) / self.tau
        if self.nogb:
            cates = self.softmax(cates_logits)
        else:
            cates_dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(1, logits=cates_logits)
            cates_sample = cates_dist.sample()
            cates_mode = self.softmax(cates_logits)
            cates = (self.is_training_ph * cates_sample +
                     (1 - self.is_training_ph) * cates_mode)
        ## ============ 得到cates，cates是近似于one-hot的向量，代表物品的分类

        # 预测表征符合的正态分布，并从中采样得到对应于每个macro concept的子表征
        z_list = []
        probs, kl = None, None
        for k in range(self.kfac):
            cates_k = torch.reshape(cates[:, k], (1, -1))

            # q-network
            x_k = (self.input_ph * cates_k)  ## 提取出所有属于类别k的物品，用来计算对应于k的子表征
            mu_k, std_k, kl_k = self.q_graph_k(x_k)
            epsilon = torch.empty(std_k.shape).to(self.device)
            nn.init.normal_(epsilon)
            z_k = mu_k + self.is_training_ph * epsilon * std_k
            kl = (kl_k if (kl is None) else (kl + kl_k))
            z_list.append(z_k)
        # 将预测得到的子表征输入以DisenGraph层为主体的残差网络
        z = torch.cat(z_list, dim=1)
        z = self.residual(z, neighbors)
        dfac = int(z.shape[1] / self.kfac)

        # 将学习了社交信息的表征送进Decoder解码器
        for k in range(self.kfac):
            cates_k = torch.reshape(cates[:, k], (1, -1))
            # p-network
            z_k = z[:,dfac * k: dfac * (k+1)]
            z_k = self.l2_normalize(z_k, axis=1)
            logits_k = torch.matmul(z_k, torch.transpose(items,0,1)) / self.tau ## 在第k个因子对应的子表征上与物品的相似度
            probs_k = torch.exp(logits_k)
            probs_k = probs_k * cates_k ## 只有物品属于类别k，才考虑相似度
            probs = (probs_k if (probs is None) else (probs + probs_k))

        # 计算网络输出结果和Loss
        logits = torch.log(probs)
        logits = self.logsoftmax(logits)
        recon_loss = torch.mean(torch.sum(-logits * self.input_ph, -1)) ## 类似于于交叉熵，self.input_ph是label

        # 计算参数的模来避免梯度爆炸 但是因为我在训练的过程中一直将rg/lam设为0，所以实际上没用到
        reg_var = torch.tensor(0.0).to(self.device)
        reg_var += torch.norm(self.items,2)
        reg_var += torch.norm(self.cores,2)
        for linear in self.linears_q:
            reg_var += torch.norm(linear.weight, 2)

        # print("recon_loss:", recon_loss, " kl:", kl)

        neg_elbo = recon_loss + self.anneal_ph * kl + self.lam * reg_var
        ## loss + KL_Divergence + regularization


        # 是否需要输出表征
        if save_emb:
            return z, logits, neg_elbo
        return logits, neg_elbo

class Scheduler:
    
    def __init__(self, mode, total_epoch, step, lambda_0):
        self.mode = mode
        self.total_epoch = total_epoch
        self.step = step
        interval = self.total_epoch / (self.step + 2)
        self.curriculum_epoch = int(interval * self.step)
        self.epoch = 0
        self.lambda_0 = lambda_0
        
    def schedule(self):
        if self.mode == 'linear':
            threshold = min(1, self.lambda_0 + (1 - self.lambda_0) * self.epoch / self.curriculum_epoch)
        elif self.mode == 'root':
            threshold = min(1, np.sqrt(self.lambda_0 ** 2 + (1 - self.lambda_0 ** 2) * self.epoch / self.curriculum_epoch))
        elif self.mode == 'geometric':
            threshold = min(1, pow(2, np.log2(self.lambda_0) - np.log2(self.lambda_0) * self.epoch / self.curriculum_epoch))
        self.epoch += 1
        return threshold

# 加载数据
# 所有数据已经被处理成：userID ItemID Rate的形式
# 用户之间的关系数据被处理为： user1ID user2ID的形式
# 且都没有多余空行
# ratingDir为训练数据 socialDir为用户关系数据 testDir为测试数据
def loadData(ratingDir, socialDir, testDir):
    with open(ratingDir) as f:
        ratings = f.readlines()
    with open(socialDir) as f:
        relates = f.readlines()

    # 加载训练集
    MaxUserId = -1
    MaxItemId = -1
    user = []
    item = []
    rate = []
    validUser = []
    validItem = []
    validRate = []
    for line in ratings:
        items = line.split()
        userId = int(items[0])
        itemId = int(items[1])
        rating = float(items[2])
        if userId > MaxUserId:
            MaxUserId = userId
        if itemId > MaxItemId:
            MaxItemId = itemId
        # 将训练数据中按照split比例分离出一部分作为验证集
        if random.random() > ARG.split:
            user.append(userId)
            item.append(itemId)
            if ARG.monorate:
                rate.append(1)
            else:
                rate.append(rating)
        else:
            validUser.append(userId)
            validItem.append(itemId)
            validRate.append(1)

    ## X为tensor，V为array
    X = torch.sparse.FloatTensor(torch.tensor([user,item]),torch.tensor(rate),torch.Size([MaxUserId+1, MaxItemId+1])).to_dense()
    V = sparse.coo_matrix((validRate, (validUser, validItem)), shape=(MaxUserId+1, MaxItemId+1)).toarray()

    # 加载用户关系数据
    edges = []
    for line in relates:
        items = line.split()
        idx1 = int(items[0])
        idx2 = int(items[1])
        edges.append((idx1, idx2))

    graph = nx.Graph()
    graph.add_nodes_from(range(MaxUserId+1))
    graph.add_edges_from(edges)

    # 加载测试集
    testUser = []
    testItem = []
    testRate = []

    with open(testDir) as f:
        test = f.readlines()

    for line in test:
        items = line.split()
        userId = int(items[0])
        itemId = int(items[1])
        testUser.append(userId)
        testItem.append(itemId)
        testRate.append(1)
    T = sparse.coo_matrix((testRate, (testUser, testItem)), shape=(MaxUserId + 1, MaxItemId + 1)).toarray()

    # 这个地方加1是为了后面用range()函数的时候可以不用再加了
    return X, graph, MaxUserId+1, MaxItemId+1, V, T

def two_k_partition(G: nx.Graph, k: int):
    if k == 0:
        return [G]
    partition_1, partition_2 = nx.algorithms.community.kernighan_lin_bisection(G)
    subgraph_1, subgraph_2 = G.subgraph(partition_1), G.subgraph(partition_2)
    if k > 1:
        subgraphs_1 = two_k_partition(subgraph_1, k - 1)
        subgraphs_2 = two_k_partition(subgraph_2, k - 1)
        subgraphs = subgraphs_1 + subgraphs_2
    else:
        subgraphs = [subgraph_1, subgraph_2]
    return subgraphs

def test(ARG, model, graph, train_data, T, dev, epoch, subgraphs):
    
    ndcg_dist = []
    r50_dist = []
    r20_dist = []
    # for edge in graph.edges():
    #     graph[edge[0]][edge[1]]["weight"] = 1
    # subgraphs = two_k_partition(graph, ARG.partitionK)
    for i in range(len(subgraphs)):
        subgraph = subgraphs[i]
        subgraph_nodes = list(subgraph.nodes())
        t = sparse.coo_matrix(T[subgraph.nodes()])
        # 用于测试的对应训练数据
        x = train_data[subgraph_nodes].to(dev)
        # 初始化子图和图采样
        mapping = dict(zip(subgraph_nodes, range(subgraph.number_of_nodes())))
        subgraph = nx.relabel_nodes(subgraph, mapping)
        neibSampler = NeibSampler(subgraph, ARG.nbsz)
        neibSampler.to(dev)
        # 得到测试结果
        model.eval()
        z, logits, loss = model(True, neibSampler.sample(), x, 1, ARG.beta)
        correlation = np.absolute(np.corrcoef(z.cpu().detach().numpy(), rowvar=False))
        # print(correlation.shape)
        plt.matshow(correlation, cmap=plt.get_cmap('Greens'))
        plt.savefig('corr_result/' + str(ARG.data) + '/' + str(epoch) + '.jpg')



def train(ARG):
    # 设置训练所用的gpu
    # torch.cuda.set_device(0)
    torch.cuda.set_device(int(ARG.gpudevice))
    use_cuda = torch.cuda.is_available() and not ARG.cpu
    dev = torch.device('cuda' if use_cuda else 'cpu')
    ARG.device = dev

    #加载训练数据，用户关系图，用户数量，物品数量，验证集，测试集
    train_data, graph, n, m, V, T \
        = loadData(ARG.data+'/train.txt', ARG.data+'/trusts.txt', ARG.data+'/test+.txt')
    cur_scheduler = Scheduler("linear", ARG.epoch, 6, 0.10)

    model = myVAE(m, ARG).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=ARG.lr, weight_decay=ARG.rg)
    print("n = ", n)
    print("edges = ", graph.number_of_edges())
    punish = float(1/160)
    for edge in graph.edges():
        graph[edge[0]][edge[1]]["weight"] = 1
    for epochNum in range(ARG.epoch):
        loss_list = []
        subgraphs = two_k_partition(graph, ARG.partitionK)
        threshold = cur_scheduler.schedule() * len(subgraphs)
        res_subgraphs = []
        for subgraph in subgraphs:
            _, _, I, _, _, _ = calc_difficulty(subgraph)
            res_subgraphs.append((subgraph, I))
        res_subgraphs.sort(key=operator.itemgetter(1))    
        
        mask = np.ones(int(threshold))
        for i in range(int(threshold), len(subgraphs)):
            mask = np.append(mask, 1 - res_subgraphs[i][1])

        cnt = 0
        z_list = []
        for subgraph, _ in res_subgraphs:
            for edge in subgraph.edges():
                graph[edge[0]][edge[1]]["weight"] += punish
            subgraph_nodes = list(subgraph.nodes())
            x = train_data[subgraph_nodes].to(dev)
            mapping = dict(zip(subgraph_nodes, range(subgraph.number_of_nodes()))) 
            subgraph = nx.relabel_nodes(subgraph, mapping)
            # 初始化对应于该子图的采样
            neibSampler = NeibSampler(subgraph, ARG.nbsz)
            neibSampler.to(dev)
            # 训练
            model.train()
            optimizer.zero_grad()
            z, logits, loss = model(True, neibSampler.sample(), x, 1, ARG.beta)
            z_list.append(z.cpu().detach().numpy())
            loss = loss * mask[cnt]
            cnt += 1
            loss.backward()
            optimizer.step()
            l = loss.item()
            loss_list.append(l)
        # print(len(z_list))
        z_list = np.concatenate(z_list, axis=0)
        print(z_list.shape)
        loss = np.mean(loss_list)
        print('Epoch ', epochNum, ' trn-loss: %.4f' % loss)
        # python sup/main_corr.py --data lastfm --epoch 10 --numGCNLayer 2 --numLayer 3 --ratio 4 --routit 10 --partitionK 0 --kfac 8 --dfac 8
        if (epochNum + 1) % 2 == 0:
            # test(ARG, model, graph, train_data, T, dev, epochNum, subgraphs)
            correlation = np.absolute(np.corrcoef(z_list, rowvar=False))
            figure = plt.figure()
            axes = figure.add_subplot(111)
            caxes = axes.matshow(correlation, interpolation ='nearest', cmap=plt.get_cmap('Greens'))
            figure.colorbar(caxes)
            plt.savefig('corr_result/' + str(ARG.data) + '/' + str(epochNum) + '.jpg')
            plt.clf()

    # 训练完毕在测试集上进行测试
    torch.save(model.state_dict(), './model/node_'+ str(ARG.data) + '_' + str(time.time()) +'.pt')
    # state_dict = torch.load('./model/node_'+ str(ARG.data) + '_' + str(best_epoch)+'.pt')
    # model.load_state_dict(state_dict)

    print(ARG.numGCNLayer, ARG.numLayer, ARG.ratio)

if __name__ == '__main__':
    ARG = parseArgs()
    train(ARG)