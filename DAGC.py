import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from B_code.utils import load_data, load_graph, normalize_adj, numpy_to_torch
from B_code.GNN_previous import GNNLayer
from B_code.eva_previous import eva
from datetime import datetime
import time
# import scipy.io as scio

tic = time.time()
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_z2 = F.relu(self.enc_1(x))
        enc_z3 = F.relu(self.enc_2(enc_z2))
        enc_z4 = F.relu(self.enc_3(enc_z3))
        z = self.z_layer(enc_z4)

        dec_z2 = F.relu(self.dec_1(z))
        dec_z3 = F.relu(self.dec_2(dec_z2))
        dec_z4 = F.relu(self.dec_3(dec_z3))
        x_bar = self.x_bar_layer(dec_z4)

        return x_bar, enc_z2, enc_z3, enc_z4, z

class MLP_L(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)

    def forward(self, mlp_in):

        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)
        return weight_output

class MLP_1(nn.Module):
    
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp,2)

    def forward(self, mlp_in):

        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1) 
        return weight_output

class MLP_2(nn.Module):

    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)

    def forward(self, mlp_in):

        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)
        return weight_output

class MLP_3(nn.Module):
    
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)

    def forward(self, mlp_in):

        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)  
        return weight_output

class MLP_ZQ(nn.Module):
    
    def __init__(self, n_mlp):
        super(MLP_ZQ, self).__init__()
        self.w_ZQ = Linear(n_mlp, 2)

    def forward(self, mlp_in):

        weight_output = F.softmax(F.leaky_relu(self.w_ZQ(mlp_in)), dim=1)  
        return weight_output

class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(SDCN, self).__init__()

        # autoencoder
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        if args.name == 'amap' or args.name == 'pubmed':
            pretrained_dict = torch.load(args.pretrain_path, map_location='cpu')
            model_dict = self.ae.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.ae.load_state_dict(model_dict)
        else:
            self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        ## 3️⃣
        self.agnn_0 = GNNLayer(n_input, n_enc_1)
        self.agnn_1 = GNNLayer(n_enc_1, n_enc_2)
        self.agnn_2 = GNNLayer(n_enc_2, n_enc_3)
        self.agnn_3 = GNNLayer(n_enc_3, n_z)
        self.agnn_z = GNNLayer(3020,n_clusters)

        self.mlp = MLP_L(3020) 

        # attention on [z_i, h_i]
        self.mlp1 = MLP_1(2*n_enc_1)
        self.mlp2 = MLP_2(2*n_enc_2)
        self.mlp3 = MLP_3(2*n_enc_3)

        self.mlp_ZQ = MLP_ZQ(2*n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.v = v

        self.n_clusters = n_clusters

    def forward(self, x, adj):
        # DAE Module
        x_bar, h1, h2, h3, z = self.ae(x)

        x_array = list(np.shape(x))
        n_x = x_array[0]

        # 5️⃣ GCN[p_1*Z + p_2*H]
        # z1
        z1 = self.agnn_0(x, adj)
        # z2
        p1 = self.mlp1( torch.cat((h1,z1), 1) )
        p1 = F.normalize(p1,p=2)
        p11 = torch.reshape(p1[:,0], [n_x, 1])
        p12 = torch.reshape(p1[:,1], [n_x, 1])
        p11_broadcast =  p11.repeat(1,500)
        p12_broadcast = p12.repeat(1,500)
        z2 = self.agnn_1( p11_broadcast.mul(z1)+p12_broadcast.mul(h1), adj)
        # z3
        p2 = self.mlp2( torch.cat((h2,z2),1) )     
        p2 = F.normalize(p2,p=2)
        p21 = torch.reshape(p2[:,0], [n_x, 1])
        p22 = torch.reshape(p2[:,1], [n_x, 1])
        p21_broadcast = p21.repeat(1,500)
        p22_broadcast = p22.repeat(1,500)
        z3 = self.agnn_2( p21_broadcast.mul(z2)+p22_broadcast.mul(h2), adj)
        # z4
        p3 = self.mlp3( torch.cat((h3,z3),1) )# self.mlp3(h2)      
        p3 = F.normalize(p3,p=2)
        p31 = torch.reshape(p3[:,0], [n_x, 1])
        p32 = torch.reshape(p3[:,1], [n_x, 1])
        p31_broadcast = p31.repeat(1,2000)
        p32_broadcast = p32.repeat(1,2000)
        z4 = self.agnn_3( p31_broadcast.mul(z3)+p32_broadcast.mul(h3), adj)

        w  = self.mlp(torch.cat((z1,z2,z3,z4,z),1))
        w = F.normalize(w,p=2)

        w0 = torch.reshape(w[:,0], [n_x, 1])
        w1 = torch.reshape(w[:,1], [n_x, 1])
        w2 = torch.reshape(w[:,2], [n_x, 1])
        w3 = torch.reshape(w[:,3], [n_x, 1])
        w4 = torch.reshape(w[:,4], [n_x, 1])

        # 2️⃣ [Z+H]
        tile_w0 = w0.repeat(1,500)
        tile_w1 = w1.repeat(1,500)
        tile_w2 = w2.repeat(1,2000)
        tile_w3 = w3.repeat(1,10)
        tile_w4 = w4.repeat(1,10)

        # 2️⃣ concat
        net_output = torch.cat((tile_w0.mul(z1), tile_w1.mul(z2), tile_w2.mul(z3), tile_w3.mul(z4), tile_w4.mul(z)), 1 )
        net_output = self.agnn_z(net_output, adj, active=False) 

        predict = F.softmax(net_output, dim=1)

        # Dual Self-supervision
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        p_ZH = self.mlp_ZQ( torch.cat((predict,q),1) )   

        p_ZH = F.normalize(p_ZH,p=2)

        p_ZH1 = torch.reshape(p_ZH[:,0], [n_x, 1])
        p_ZH2 = torch.reshape(p_ZH[:,1], [n_x, 1])
        p_ZH1_broadcast = p_ZH1.repeat(1,self.n_clusters)
        p_ZH2_broadcast = p_ZH2.repeat(1,self.n_clusters)
        z_F = p_ZH1_broadcast.mul(predict)+p_ZH2_broadcast.mul(q)
        z_F = F.softmax(z_F, dim=1)

        # # 🟡pseudo_label_loss
        clu_assignment = torch.argmax(z_F, -1)
        clu_assignment_onehot=F.one_hot(clu_assignment, self.n_clusters)
        thres = 0.8
        thres_matrix = torch.zeros_like(z_F) + thres
        weight_label = torch.ge(F.normalize(z_F,p=2), thres_matrix).type(torch.cuda.FloatTensor)
        pseudo_label_loss = BCE(z_F, clu_assignment_onehot, weight_label)
        return x_bar, q, predict, z, net_output, pseudo_label_loss, z_F

def BCE(out, tar, weight):
    eps = 1e-12 # The case without eps could lead to the `nan' situation
    l_n = weight * ( tar * (torch.log(out+eps)) + (1 - tar) * (torch.log(1 - out+eps)) )
    l = -torch.sum(l_n) / torch.numel(out)
    return l

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def num_net_parameter(net):
    all_num = sum(i.numel() for i in net.parameters())
    print ('[The network parameters]', all_num)

def train_sdcn(dataset):
    # 🎏
    dataname = args.name
    file_out = open('./R_output/'+dataname+'_results.out', 'a') 
    print("The experimental results", file=file_out)

    # lambda_1 = [0.1] # [0.001,0.01,0.1,1,10,100,1000] # 
    # lambda_2 = [0.1] # [0.001,0.01,0.1,1,10,100,1000] # 
    # lambda_3 = [0.001] # [0.001,0.01,0.1,1,10,100,1000] # 
    # for ld1 in lambda_1:
    #     for ld2 in lambda_2:
    #         for ld3 in lambda_3:

    ld1 = args.ld1
    ld2 = args.ld2
    ld3 = args.ld3

    print("lambda_1: ", ld1, "lambda_2: ", ld2, "lambda_3: ", ld3, file=file_out)
    model = SDCN(500, 500, 2000, 2000, 500, 500,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=args.n_clusters,
                v=1.0).cuda()#.to(device)

    print(num_net_parameter(model))

    optimizer = Adam(model.parameters(), lr=args.lr)


    # KNN Graph
    if args.name == 'amap' or args.name == 'pubmed':
        load_path = "data/" + args.name + "/" + args.name
        adj = np.load(load_path+"_adj.npy", allow_pickle=True)
        adj = normalize_adj(adj, self_loop=True, symmetry=False)
        adj = numpy_to_torch(adj, sparse=True).to(torch.device("cuda")) # opt.args.device = torch.device("cuda" if opt.args.cuda else "cpu")
    else:
        adj = load_graph(args.name, args.k)
        adj = adj.cuda()#.to(device)

    data = torch.Tensor(dataset.x).cuda()#.to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)


    iters10_kmeans_iter_F = []
    iters10_NMI_iter_F = []
    iters10_ARI_iter_F = []
    iters10_F1_iter_F = []

    z_1st = z
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z_1st.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()#.to(device)
    acc,nmi,ari,f1 = eva(y, y_pred, 'pae')

    kmeans_iter_F = []
    NMI_iter_F = []
    ARI_iter_F = []
    F1_iter_F = []

    for epoch in range(200):

        if epoch % 1 == 0:
            _, tmp_q, pred, _, _, _, z_F = model(data, adj)
            p = target_distribution(pred.data)
            res4 = z_F.data.cpu().numpy().argmax(1) 
            acc,nmi,ari,f1 = eva(y, res4, str(epoch) + 'F')
            kmeans_iter_F.append(acc)
            NMI_iter_F.append(nmi)
            ARI_iter_F.append(ari)
            F1_iter_F.append(f1)

        x_bar, q, pred, _, _, pl_loss, z_F = model(data, adj)

        KL_QP = F.kl_div(q.log(), p, reduction='batchmean')
        KL_ZP = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        loss = re_loss \
            + ld1 * KL_QP + ld1 * KL_ZP \
            + ld2 * (F.kl_div(q.log(), pred, reduction='batchmean')) \
            + ld3 * pl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # _F
    kmeans_max= np.max(kmeans_iter_F)
    nmi_max= np.max(NMI_iter_F)
    ari_max= np.max(ARI_iter_F)
    F1_max= np.max(F1_iter_F)
    iters10_kmeans_iter_F.append(round(kmeans_max,5))
    iters10_NMI_iter_F.append(round(nmi_max,5))
    iters10_ARI_iter_F.append(round(ari_max,5))
    iters10_F1_iter_F.append(round(F1_max,5))

    print("#################"+dataname+"####################", file=file_out)
    print("kmeans F mean",round(np.mean(iters10_kmeans_iter_F),5),"max",np.max(iters10_kmeans_iter_F),"\n",iters10_kmeans_iter_F, file=file_out)
    print("NMI mean",round(np.mean(iters10_NMI_iter_F),5),"max",np.max(iters10_NMI_iter_F),"\n",iters10_NMI_iter_F, file=file_out)
    print("ARI mean",round(np.mean(iters10_ARI_iter_F),5),"max",np.max(iters10_ARI_iter_F),"\n",iters10_ARI_iter_F, file=file_out)
    print("F1  mean",round(np.mean(iters10_F1_iter_F),5),"max",np.max(iters10_F1_iter_F),"\n",iters10_F1_iter_F, file=file_out)
    print(':acc, nmi, ari, f1: \n{:.4f}\n{:.4f}\n{:.4f}\n{:.4f}'.format(round(np.mean(iters10_kmeans_iter_F),5),round(np.mean(iters10_NMI_iter_F),5),round(np.mean(iters10_ARI_iter_F),5),round(np.mean(iters10_F1_iter_F),5)), file=file_out)

    file_out.close()

if __name__ == "__main__":
    # 🎏 iters
    iters = 10 # 

    for iter_num in range(iters):
        print(iter_num)
        parser = argparse.ArgumentParser(
            description='train',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str, default='acm')
        parser.add_argument('--k', type=int, default=3)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--n_clusters', default=3, type=int)
        parser.add_argument('--n_z', default=10, type=int)
        parser.add_argument('--pretrain_path', type=str, default='pkl')
        args = parser.parse_args()
        args.cuda = torch.cuda.is_available()
        print("use cuda: {}".format(args.cuda))
        device = torch.device("cuda" if args.cuda else "cpu")

        args.pretrain_path = 'data/{}.pkl'.format(args.name)
        dataset = load_data(args.name)

        if args.name == 'usps':
            args.k = 3 
            args.ld1 = 0.1
            args.ld2 = 0.1
            args.ld3 = 0.001
            args.n_clusters = 10
            args.n_input = 256

        if args.name == 'hhar':
            args.k = 5
            args.ld1 = 0.1
            args.ld2 = 0.1
            args.ld3 = 0.01
            args.n_clusters = 6
            args.n_input = 561

        if args.name == 'reut':
            args.k = 3
            args.ld1 = 10
            args.ld2 = 10
            args.ld3 = 100
            args.lr = 1e-4
            args.n_clusters = 4
            args.n_input = 2000

        if args.name == 'acm':
            args.k = None
            args.ld1 = 0.1
            args.ld2 = 0.1
            args.ld3 = 0.001
            args.n_clusters = 3
            args.n_input = 1870

        if args.name == 'dblp':
            args.k = None
            args.ld1 = 1
            args.ld2 = 1
            args.ld3 = 10
            args.n_clusters = 4
            args.n_input = 334

        if args.name == 'cite':
            args.lr = 1e-4
            args.k = None
            args.ld1 = 100
            args.ld2 = 100
            args.ld3 = 0.001
            args.n_clusters = 6
            args.n_input = 3703

        if args.name == 'amap':
            args.lr = 1e-4
            args.k = None
            args.ld1 = 0.001
            args.ld2 = 1000
            args.ld3 = 0.01
            args.n_clusters = 8
            args.n_input = 745
            
        if args.name == 'pubmed':
            args.lr = 1e-3
            args.k = None
            args.ld1 = 0.01
            args.ld2 = 0.01
            args.ld3 = 1
            args.n_clusters = 3
            args.n_input = 500

        if args.name == 'AIDS':
            args.lr = 1e-4
            args.k = None
            args.ld1 = 1000
            args.ld2 = 1000
            args.ld3 = 0.001
            args.n_clusters = 38
            args.n_input = 4

        print(args)
        train_sdcn(dataset)

    toc = time.time()
    print("Time:", (toc - tic))