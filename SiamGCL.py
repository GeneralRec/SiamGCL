import torch
import torch.nn as nn
import torch.nn.functional as F
from base.graph_recommender import GraphRecommender
from util.conf import OptionConf
from base.torch_interface import TorchGraphInterface
from random import shuffle,choice
import csv

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
class SiamGCL(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SiamGCL, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SiamGCL'])
        self.cl_rate = float(args['-lambda'])
        self.alpha = float(args['-alpha'])
        self.gamma = float(args['-gamma'])
        self.tau = float(args['-tau'])
        self.n_layers = int(args['-n_layer'])
        self.model = SiamGCL_Encoder(self.data, self.emb_size, self.alpha, self.n_layers)
        self.epoch_losses = []
        self.epoch_recalls = []
        self.epoch_ndcgs = []
        self.epoch_hits = []
        self.epoch_precisions = []

    def train(self):
        model = self.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        self.pos_pairs = []
        for epoch in range(self.maxEpoch):
            cumulative_loss = 0  # Initialize cumulative loss for this epoch
            num_batches = 0  # Counter for number of batches

            for n, batch in enumerate(self.next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                self.pos_pairs.extend(zip(user_idx, pos_idx))
                input1 = (user_idx, pos_idx, neg_idx, True)
                input2 = (user_idx, pos_idx, neg_idx, True)
                rec_user_emb1, rec_item_emb1,rec_user_emb2, rec_item_emb2 = model(input1,input2)

                user_emb1, pos_item_emb1, neg_item_emb1 = rec_user_emb1[user_idx], rec_item_emb1[pos_idx], rec_item_emb1[neg_idx]
                rec_loss1 = self.bpr_loss(user_emb1, pos_item_emb1, neg_item_emb1)
                cl_loss1 = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx])
                aligh_loss1 = self.alignment_loss(user_emb1, pos_item_emb1)
                batch_loss1 =  self.gamma * aligh_loss1 + rec_loss1 + self.l2_reg_loss(self.reg, user_emb1, pos_item_emb1) + cl_loss1

                user_emb2, pos_item_emb2, neg_item_emb2 = rec_user_emb2[user_idx], rec_item_emb2[pos_idx], rec_item_emb2[neg_idx]
                rec_loss2 = self.bpr_loss(user_emb2, pos_item_emb2, neg_item_emb2)
                cl_loss2 = self.cl_rate * self.cal_cl_loss([user_idx, pos_idx])
                aligh_loss2 = self.alignment_loss(user_emb2, pos_item_emb2)
                batch_loss2 = self.gamma * aligh_loss2 + rec_loss2 + self.l2_reg_loss(self.reg, user_emb2, pos_item_emb2) + cl_loss2

                batch_loss = (batch_loss1 + batch_loss2)/2
                rec_loss = (rec_loss1 + rec_loss2)/2
                cl_loss = (cl_loss1 + cl_loss2)/2

                # Accumulate the batch loss
                cumulative_loss += rec_loss.item()
                num_batches += 1

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0  and n > 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())



            average_loss_for_epoch = cumulative_loss / num_batches
            self.epoch_losses.append(average_loss_for_epoch)

            #Save the data at the end of each epoch
            #self.save_to_csv('SiamGCL_loss_results_gowalla.csv', "loss")


            with torch.no_grad():
                user_indices = torch.arange(self.data.user_num).to(device)
                item_indices = torch.arange(self.data.item_num).to(device)
                self.user_emb, self.item_emb = model.forward_once(user_indices, item_indices, item_indices, False)

            a = self.fast_evaluation(epoch)

            self.epoch_hits.append(a[0][10:])
            self.epoch_precisions.append(a[1][10:])

            self.epoch_recalls.append(a[2][7:])
            self.epoch_ndcgs.append(a[3][5:])

            # self.save_to_csv('SiamGCL_hit_results_gowalla.csv', "hit")
            # self.save_to_csv('SiamGCL_precision_results_gowalla.csv', "precision")
            # self.save_to_csv('SiamGCL_recall_results_gowalla.csv', "recall")
            # self.save_to_csv('SiamGCL_ndcg_results_gowalla.csv', "ndcg")

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
        # 在训练结束后
        # self.save_embeddings('SiamGCL_user_embeddings_gowalla.pt', 'SiamGCL_item_embeddings_gowalla.pt')
        # self.save_pos_pairs('SiamGCL_positive_pairs_gowalla.csv')

    def save_embeddings(self, user_embedding_path, item_embedding_path):
        torch.save(self.best_user_emb, user_embedding_path)
        torch.save(self.best_item_emb, item_embedding_path)

    def save_pos_pairs(self, path):
        with open(path, 'w') as f:
            for u, i in self.pos_pairs:
                f.write(f"{u},{i}\n")

    def InfoNCE(self, view1, view2, temperature, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score + 10e-6)
        return torch.mean(cl_loss)

    def bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):
        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)

    def l2_reg_loss(self, reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2) / emb.shape[0]
        return emb_loss * reg

    def alignment(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def alignment_loss(self, user_emb, item_emb):
        align = self.alignment(user_emb, item_emb)
        return align
    def cal_cl_loss(self, idx):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).to(device)
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).to(device)
        user_indices = torch.arange(self.data.user_num).to(device)
        item_indices = torch.arange(self.data.item_num).to(device)

        user_view_1, item_view_1 = self.model.forward_once(user_indices,item_indices,item_indices,perturbed=True)
        user_view_2, item_view_2 = self.model.forward_once(user_indices,item_indices,item_indices,perturbed=True)
        user_cl_loss = self.InfoNCE(user_view_1[u_idx], user_view_2[u_idx], self.tau)
        item_cl_loss = self.InfoNCE(item_view_1[i_idx], item_view_2[i_idx], self.tau)
        return user_cl_loss + item_cl_loss

    def save_to_csv(self, filename, flag):
        with open(filename, 'w', newline='') as csvfile:
            if flag == "loss":
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Epoch', 'Loss'])
                for epoch, metric in enumerate(self.epoch_losses):
                    csv_writer.writerow([epoch+1, metric])

            if flag == "hit":
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Epoch', 'Hit'])
                for epoch, metric in enumerate(self.epoch_hits):
                    csv_writer.writerow([epoch+1, metric])

            if flag == "precision":
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Epoch', 'Precision'])
                for epoch, metric in enumerate(self.epoch_precisions):
                    csv_writer.writerow([epoch+1, metric])

            if flag == "recall":
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Epoch', 'Recall'])
                for epoch, metric in enumerate(self.epoch_recalls):
                    csv_writer.writerow([epoch+1, metric])

            if flag == "ndcg":
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Epoch', 'NDCG'])
                for epoch, metric in enumerate(self.epoch_ndcgs):
                    csv_writer.writerow([epoch+1, metric])

    def next_batch_pairwise(self, data, batch_size, n_negs=1):
        training_data = data.training_data
        shuffle(training_data)
        ptr = 0
        data_size = len(training_data)
        while ptr < data_size:
            if ptr + batch_size < data_size:
                batch_end = ptr + batch_size
            else:
                batch_end = data_size
            users = [training_data[idx][0] for idx in range(ptr, batch_end)]
            items = [training_data[idx][1] for idx in range(ptr, batch_end)]
            ptr = batch_end
            u_idx, i_idx, j_idx = [], [], []
            item_list = list(data.item.keys())
            for i, user in enumerate(users):
                i_idx.append(data.item[items[i]])
                u_idx.append(data.user[user])
                for m in range(n_negs):
                    neg_item = choice(item_list)
                    while neg_item in data.training_set_u[user]:
                        neg_item = choice(item_list)
                    j_idx.append(data.item[neg_item])
            yield u_idx, i_idx, j_idx

    def save(self):
        with torch.no_grad():
            user_indices = torch.arange(self.data.user_num).to(device)
            item_indices = torch.arange(self.data.item_num).to(device)
            self.best_user_emb, self.best_item_emb = self.model.forward_once(user_indices,item_indices,item_indices,False)

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class EU(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.5):
        super(EU, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return F.normalize(self.layer(x), dim=-1)


class SiamGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, alpha, n_layers):
        super(SiamGCL_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.alpha = alpha
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).to(device)
        self.noise_generator_user = EU(self.emb_size, self.emb_size).to(device)
        self.noise_generator_item = EU(self.emb_size, self.emb_size).to(device)


    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward_once(self, user_idx, pos_idx, neg_idx, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []

        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)

            if perturbed:

                noise_user = self.noise_generator_user(ego_embeddings[:self.data.user_num])
                noise_item = self.noise_generator_item(ego_embeddings[self.data.user_num:])
                noise = torch.cat([noise_user, noise_item], 0)
                ego_embeddings = ego_embeddings + torch.sign(ego_embeddings) * noise * self.alpha

            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])

        return user_all_embeddings, item_all_embeddings

    def forward(self, input1, input2):
        output1_user, output1_item = self.forward_once(*input1)
        output2_user, output2_item = self.forward_once(*input2)

        return output1_user, output1_item, output2_user, output2_item

