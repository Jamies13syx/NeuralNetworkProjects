import torch
import torch.nn as nn

rnn_size = 100
rnn_nLayers = 2
layer0 = 100
layer1 = 100

use_cuda = False
print(torch.version.__version__, use_cuda)

learning_rate = 0.001
L2_lambda = 0.0
nEpochs = 25
dropout = 0.0

charDict = dict()  # for convert data into ndx_data


class RNN(nn.Module):

    def __init__(self, specs):
        super(RNN, self).__init__()

        nChars, embed_size, rnn_layers, ffnn_layers, dropout = specs

        self.CharEmbed = nn.Embedding(nChars, embed_size)
        self.rnn = nn.GRU(embed_size, rnn_size, rnn_nLayers, dropout=dropout, batch_first=True)

        self.layers = nn.ModuleList([])
        prev_size = rnn_size

        for i, layer_size in enumerate(ffnn_layers):
            layer = nn.Linear(prev_size, layer_size)
            self.layers.append(layer)
            prev_size = layer_size

        self.non_linear = nn.LeakyReLU(negative_slope=0.01)

        self.dropout = nn.Dropout(dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
                pass

    def forward(self, character, hidden=None):
        # print(character)
        character = character.view(1, 1)
        embed = self.CharEmbed(character)

        prev, hidden = self.rnn(embed, hidden)

        for layer in self.layers:
            prev = layer(prev)
            prev = self.non_linear(prev)
            prev = self.dropout(prev)

        self.out = nn.Linear(100, 1)  # output floating number stream

        out = self.out(prev)
        out = out.squeeze()

        return out


def RNN_train(model, optimizer, criterion, input, target, update=True):

    target = torch.tensor(target, dtype=torch.float)

    model.zero_grad()
    loss = 0

    out = model(input)

    loss += criterion(out, target)

    if update:
        if not loss is 0:
            loss.backward()
            optimizer.step()

    return loss.data.item()


def train_rnn_model(model, training_data, charDict):
    if use_cuda:
        model = model.cuda()

    # define the loss functions
    criterion = nn.BCEWithLogitsLoss()

    # choose an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)

    for e in range(nEpochs):
        for data in training_data:
            targets = []
            for i in range(len(data) - 1):
                if data[i] != " " and data[i + 1] != " ":
                    targets.append(0)
                elif data[i] != " " and data[i + 1] == " ":
                    targets.append(1)
                elif data[i] == " ":
                    pass
            inputs = data.replace(" ", "").strip()
            for i, character in enumerate(inputs):
                ndx_data = torch.tensor(charDict[character], dtype=torch.long)
                train_loss = 0
                model.train()
                loss = RNN_train(model, optimizer, criterion, ndx_data, targets[i], update=True)
                train_loss += loss
                print(train_loss)

    return model

def count():
    training_data = open("./seg_data/msr_training.utf8", 'r', encoding="utf-8")
    for data in training_data:
        for i in range(len(data)):
            if data[i] in charDict.keys():
                pass
            else:
                charDict[data[i]] = len(charDict)
    return len(charDict)


def segmentation(out, data):
    output = str()
    tuning = 1.10
    for i in range(len(out)):
        out[i] = abs(out[i])
    avg_distance = sum(out) / len(out)
    for i in range(len(out)):
        if out[i] < avg_distance * tuning:
            out[i] = 0
        else:
            out[i] = 1

    for j in range(len(data)):
        if out[j] == 0:
            output = output + data[j]
        else:
            output = output + data[j] + " "
    return output


if __name__ == '__main__':
    char_embed_size = 10
    RNN_layers = [rnn_size, rnn_nLayers]
    FFNN_layers = [layer0, layer1]

    training_data = open("./seg_data/msr_training.utf8", 'r', encoding="utf-8")

    nChars = count()
    print(nChars)

    specs = [nChars, char_embed_size, RNN_layers, FFNN_layers, dropout]
    model = RNN(specs)
    # try:
    #     train_rnn_model(model, training_data, charDict)
    # except Exception:
    #     pass

    test_data = open("./seg_data/msr_test.utf8", 'r', encoding="utf-8")
    for data in test_data:
        for i in range(len(data)):
            if data[i] in charDict.keys():
                test_ndx = torch.tensor(charDict[data[i]], dtype=torch.long)
            else:
                test_ndx = 0
        out = model(test_ndx)
        print(out)
        # out = out.tolist()
        # print(segmentation(out, data))
