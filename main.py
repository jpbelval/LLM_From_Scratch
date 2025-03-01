import re
from importlib.metadata import version
from dataset import GPTDatasetV1, create_dataloader_v1
import torch

max_length = 4
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Dataloader

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# Embeddings

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 50257
output_dim = 256
torch.manual_seed(123)
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings)

# Attention scores
query = input_embeddings[0][1]
atn_scores_2 = torch.empty(input_embeddings[0].shape[0])
for i, x_i in enumerate(input_embeddings[0]):
    atn_scores_2[i] = torch.dot(x_i, query)
print(atn_scores_2)
attn_weights_2 = torch.softmax(atn_scores_2, dim=0)
print("Attention weights: ", attn_weights_2)
print("Sum: ", attn_weights_2.sum())
