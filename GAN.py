import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.nn.pytorch import *
from torch.utils.data import DataLoader

from .dataset import Dataset


class GANLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        alpha=0.2,
        agg_activation=F.elu,
    ):
        super(GANLayer, self).__init__()

        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(num_heads, out_dim, 1)))
        self.attn_drop = nn.Dropout(attn_drop)
        self.activation = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax

        self.agg_activation = agg_activation

    def clean_data(self):
        ndata_names = ["ft", "a1", "a2"]
        edata_names = ["a_drop"]
        for name in ndata_names:
            self.g.ndata.pop(name)
        for name in edata_names:
            self.g.edata.pop(name)

    def forward(self, feat, bg):
        self.g = bg
        h = self.feat_drop(feat)
        ft = self.fc(h).reshape((h.shape[0], self.num_heads, -1))
        head_ft = ft.transpose(0, 1)
        a1 = torch.bmm(head_ft, self.attn_l).transpose(0, 1)  # V x K x 1
        a2 = torch.bmm(head_ft, self.attn_r).transpose(0, 1)  # V x K x 1
        self.g.ndata.update({"ft": ft, "a1": a1, "a2": a2})
        self.g.apply_edges(self.edge_attention)
        self.edge_softmax()
        self.g.update_all(fn.src_mul_edge("ft", "a_drop", "ft"), fn.sum("ft", "ft"))
        ret = self.g.ndata["ft"]
        ret = ret.flatten(1)

        if self.agg_activation is not None:
            ret = self.agg_activation(ret)

        self.clean_data()

        return ret

    def edge_attention(self, edges):
        a = self.activation(edges.src["a1"] + edges.dst["a2"])
        return {"a": a}

    def edge_softmax(self):
        attention = self.softmax(self.g, self.g.edata.pop("a"))
        self.g.edata["a_drop"] = self.attn_drop(attention)


class GAN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads, n_classes):
        super(GAN, self).__init__()

        self.layers = nn.ModuleList(
            [
                GANLayer(in_dim, hidden_dim, num_heads),
                GANLayer(hidden_dim * num_heads, hidden_dim, num_heads),
            ]
        )
        self.classify = nn.Linear(hidden_dim * num_heads, n_classes)

    def forward(self, bg):
        h = bg.ndata["features"]
        for _, gnn in enumerate(self.layers):
            h = gnn(h, bg)
        bg.ndata["features"] = h
        hg = dgl.mean_nodes(bg, "features")
        return self.classify(hg)


def main():
    trainset = Dataset(
        is_training=True,
    )
    testset = Dataset(is_training=False)

    def aggregate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)

    data_loader = DataLoader(
        trainset, batch_size=100, shuffle=True, collate_fn=aggregate
    )

    def evaluate():
        model.eval()
        test_X, test_Y = map(list, zip(*testset))
        test_bg = dgl.batch(test_X)
        test_Y = torch.tensor(test_Y).float().view(-1, 1)
        model_output = model(test_bg)
        probs_Y = torch.softmax(model_output, 1)
        sampled_Y = torch.multinomial(probs_Y, 1)
        argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
        print(
            "Test set sampled predictions accuracy: {:.4f}%".format(
                (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100
            )
        )
        print(
            "Test set argmax predictions accuracy: {:4f}%".format(
                (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100
            )
        )

    model = GAN(200, 20, 10, trainset.num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    model.train()

    epoch_losses = []
    for epoch in range(20):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= iter + 1
        print("Epoch {}, loss {:.4f}".format(epoch, epoch_loss))
        if epoch % 5 == 0:
            evaluate()
        epoch_losses.append(epoch_loss)

    evaluate()


if __name__ == "__main__":
    main()
