import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import scipy.io as sio

def load_graph(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k) 
    else:
        path = 'graph/{}_graph.txt'.format(dataset) 
    if dataset == 'cite' or dataset =='hhar' or dataset =='reut':
        data_cite = sio.loadmat('./data/{}.mat'.format(dataset))
        data = data_cite['fea']
    else:
        data = np.loadtxt('data/{}.txt'.format(dataset))
        data = np.around(data,6)
    if dataset == 'AIDS':
        load_path = "./data/" + dataset + "/" + dataset
        adj = sp.load_npz(load_path+"_adj.npz")
    else:
        n, _ = data.shape
        idx = np.array([i for i in range(n)], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(path, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),dtype=np.int32).reshape(edges_unordered.shape)
        x1 = np.ones(edges.shape[0])
        x2 = (edges[:, 0], edges[:, 1])
        adj = sp.coo_matrix((x1, x2),
                            shape=(n, n), dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])
        adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def adj_norm(adj):
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def normalize_adj(adj, self_loop=True, symmetry=False):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj
    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)
    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), adj_tmp)
    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj_tmp)
    return norm_adj

def numpy_to_torch(a, sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param sparse: is sparse tensor or not
    :return: torch tensor
    """
    if sparse:
        a = torch.sparse.Tensor(a)
        a = a.to_sparse()
    else:
        a = torch.FloatTensor(a)
    return a

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class load_data(Dataset):
    def __init__(self, dataset):

        if dataset == 'cite' or dataset =='hhar' or dataset =='reut' \
            or dataset =='dblp_for_np' or dataset =='acm_for_np'or dataset =='usps_for_np'or dataset =='reut_for_np':
            data_cite = sio.loadmat('./data/{}.mat'.format(dataset))
            self.x = np.array(data_cite['fea'])
            self.x.astype(np.float64)
            self.y = np.array(data_cite['gnd'])
            self.y.astype(np.int64)
            self.y = self.y[:,-1]
        elif dataset == 'amap' or dataset == 'pubmed':
            load_path = "data/" + dataset + "/" + dataset
            self.x = np.load(load_path+"_feat.npy", allow_pickle=True)
            self.y = np.load(load_path+"_label.npy", allow_pickle=True)
        else:
            self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
            self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))
