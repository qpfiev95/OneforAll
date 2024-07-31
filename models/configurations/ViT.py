import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_linear = nn.Linear(embed_dim, embed_dim)
        self.key_linear = nn.Linear(embed_dim, embed_dim)
        self.value_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        queries = self.query_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        weighted_values = torch.matmul(attention_probs, values).transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                                                                  embed_dim)

        output = self.output_linear(weighted_values)

        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_rate=0.0):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout_rate=0.0):
        super(ViTBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout_rate)
        self.attention_norm = nn.LayerNorm(embed_dim)

        self.feed_forward = FeedForward(embed_dim, hidden_dim, dropout_rate)
        self.feed_forward_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x

        x = self.attention_norm(x + self.attention(x))

        x = self.feed_forward_norm(x + self.feed_forward(x))

        output = residual + x

        return output


### Testing
# Set the random seed for reproducibility
torch.manual_seed(42)

# Define the input dimensions
batch_size = 4
num_channels = 3
image_height = 224
image_width = 224
embed_dim = 256

# Create random input tensors
input_tensor = torch.randn(batch_size, num_channels, image_height, image_width)

# Reshape the input tensor to match the ViT input shape
input_tensor = input_tensor.view(batch_size, -1, embed_dim)

# Instantiate the ViT block
vit_block = ViTBlock(embed_dim=embed_dim, num_heads=8, hidden_dim=512, dropout_rate=0.1)

# Pass the input through the ViT block
output_tensor = vit_block(input_tensor)

# Print the shapes of the input and output tensors
print("Input tensor shape:", input_tensor.shape)
print("Output tensor shape:", output_tensor.shape)