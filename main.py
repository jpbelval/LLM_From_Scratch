import re
from importlib.metadata import version
from dataset import GPTDatasetV1, create_dataloader_v1
from trainable_attention_weights import SelfAttention_v2
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

# Attention vectors
torch.manual_seed(123)
sa_v1 = SelfAttention_v2(d_in=256, d_out=256)
print(sa_v1(input_embeddings[0]))