import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import Dataset

msg = fn.copy_src(src="features", out="m")


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data["features"])
        h = self.activation(h)
        return {"features": h}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCNLayer, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata["features"] = feature

    def reduce(self, nodes):
        accum = torch.mean(nodes.mailbox["m"], 1)
        return {"features": accum}


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCN, self).__init__()

        self.layers = nn.ModuleList(
            [
                GCNLayer(in_dim, hidden_dim, F.relu),
                GCNLayer(hidden_dim, hidden_dim, F.relu),
                GCNLayer(hidden_dim, hidden_dim, F.relu),
            ]
        )
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata["features"]
        for conv in self.layers:
            h = conv(g, h)
        g.ndata["features"] = h
        hg = dgl.mean_nodes(g, "features")
        return self.classify(hg)


def main():
    trainset = Dataset(
        is_training=True,
    )
    testset = Dataset(
        is_training=False,
    )

    def collate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, torch.tensor(labels)

    data_loader = DataLoader(trainset, batch_size=100, shuffle=True, collate_fn=collate)

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

    model = GCN(200, 256, trainset.num_classes)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
