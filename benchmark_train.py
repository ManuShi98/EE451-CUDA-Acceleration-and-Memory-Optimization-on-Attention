import argparse

parser = argparse.ArgumentParser(description='My Program')
parser.add_argument('--attention_type', type=str, default='cpp')
args = parser.parse_args()



import numpy as np

# Load train_sentences.npy
train_sentences = np.load("./data/train_sentences.npy")

# Load test_sentences.npy
test_sentences = np.load("./data/test_sentences.npy")

# Load train_labels.npy
train_labels = np.load("./data/train_labels.npy")

# Load test_labels.npy
test_labels = np.load("./data/test_labels.npy")

import pickle

# Loading the word2idx, idx2word dictionary from a file
with open("./data/word2idx.pickle", "rb") as f:
    word2idx = pickle.load(f)

# with open("./data/idx2word.pickle", "rb") as f:
#     idx2word = pickle.load(f)


split_frac = 0.5
split_id = int(split_frac * len(test_sentences))
val_sentences, test_sentences = test_sentences[:split_id], test_sentences[split_id:]
val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]

print(f"Val len: {val_sentences.shape}, Test len: {test_sentences.shape}")


import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

train_data = TensorDataset(
    torch.from_numpy(train_sentences).float(), torch.from_numpy(train_labels)
)

val_data = TensorDataset(torch.from_numpy(val_sentences).float(), torch.from_numpy(val_labels))

test_data = TensorDataset(
    torch.from_numpy(test_sentences).float(), torch.from_numpy(test_labels)
)

batch_size = 100



train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)


# free memory
del train_sentences, test_sentences

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    
# dataiter = iter(train_loader)
# sample_x, sample_y = dataiter.next()

# print(sample_x.shape, sample_y.shape)

if args.attention_type == "cpp":
    from cpp.attention import ATTENTION
elif args.attention_type == "cuda":
    from cuda.attention import ATTENTION
elif args.attention_type == "fused":

    from fused.fused_attention import attention



class FUSED_ATTENTION(nn.Module):
    def __init__(self):
        super(FUSED_ATTENTION, self).__init__(

    def forward(self, q, k, v):
        return attention(q, k, v, q.size(-1))
        
class SentimentNet(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        hidden_dim,
        n_layers,
        drop_prob=0.5,
        attention_type = "cpp",
    ):
        super(SentimentNet, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if attention_type in ["cpp", "cuda"]:
            self.att = ATTENTION()
        elif attention_type == "fused":
            self.att = FUSED_ATTENTION()
        self.attention_type = attention_type
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.long()  # (100, 200)

        embeds = self.embedding(x)  # (100, 200, 400)

        # lstm_out: (100, 200, 512)
        # lstm_out, hidden = self.lstm(embeds, hidden)
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)  # (20000, 512)
        # out = self.dropout(lstm_out)
        batch_size, seq_len, hidden_size = embeds.size()
        embeds = embeds.view(batch_size, 8, seq_len, self.hidden_dim//8)
        if self.attention_type in ["fused"]:
            embeds = embeds.half()
        import math
        # att_out = self.att(embeds, embeds, embeds)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        att_out = self.att(embeds, embeds, embeds)

        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        print(f'Time elapsed for attention: {elapsed_time:.2f} ms')
        att_out = att_out.contiguous().view(-1, self.hidden_dim)  # (20000, 512)
        out = self.dropout(att_out)
        out = out.float()
        out = self.fc(out)
        out = self.sigmoid(out)  # (20000, 1)
        out = out.view(batch_size, -1)  # (100, 200)
        # get last timestamp output of every instances in the batch
        out = out[:, -1]  # (100, 1)
        return out

    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #     hidden = (
    #         weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
    #         weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
    #     )
    #     return hidden
        
vocab_size = len(word2idx) + 1
output_size = 1
embedding_dim = 512
hidden_dim = 512
n_layers = 1

model = SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, attention_type=args.attention_type)
model.to(device)
# model.float()
print(model)

print(vocab_size)

lr = 0.005
# Binary Cross Entropy
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 5
print_every = 100
clip = 5
valid_loss_min = np.Inf

counter = 0


model.train()

import time

start_time = time.time()
for i in range(epochs):
    # h = model.init_hidden(batch_size)

    for inputs, labels in train_loader:
        counter += 1
        # h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        # inputs.shape: (100, 200), h[0].shape: (2, 100, 512)
        output = model(inputs)

        # output.shape: torch.Size([100]), output.squeeze().shape: torch.Size([100]), labels.float().shape: torch.Size([100])
        loss = criterion(output.squeeze() if output.shape[0]!=1 else output, labels.float())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        if counter % print_every == 0:
            # val_h = model.init_hidden(batch_size)
            val_losses = []
            model.eval()
            for inp, lab in val_loader:
                # val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out = model(inp)
                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())

            model.train()
            print(
                "Epoch: {}/{}...".format(i + 1, epochs),
                "Step: {}...".format(counter),
                "Loss: {:.6f}...".format(loss.item()),
                "Val Loss: {:.6f}".format(np.mean(val_losses)),
            )
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), "./models/state_dict.pt")
                print(
                    "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                        valid_loss_min, np.mean(val_losses)
                    )
                )
                valid_loss_min = np.mean(val_losses)

end_time = time.time()
total_time = end_time - start_time
print("Total training time: {:.2f} seconds".format(total_time))

# Loading the best model
model.load_state_dict(torch.load("./models/state_dict.pt"))

test_losses = []
num_correct = 0
# h = model.init_hidden(batch_size)

model.eval()
for inputs, labels in test_loader:

    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc * 100))

                
                