'''
 * The Inference of RAM and Tag2Text Models
 * Written by Xinyu Huang
'''
import torch
import torch.nn as nn

def inference_tag2text(image, model, input_tag="None"):

    with torch.no_grad():
        caption, tag_predict = model.generate(image,
                                              tag_input=None,
                                              max_length=50,
                                              return_tag_predict=True)

    if input_tag == '' or input_tag == 'none' or input_tag == 'None':
        return tag_predict[0], None, caption[0]

    # If user input specified tags:
    else:
        input_tag_list = []
        input_tag_list.append(input_tag.replace(',', ' | '))

        with torch.no_grad():
            caption, input_tag = model.generate(image,
                                                tag_input=input_tag_list,
                                                max_length=50,
                                                return_tag_predict=True)

        return tag_predict[0], input_tag[0], caption[0]


def inference_ram(image, model):

    with torch.no_grad():
        tags, tags_chinese = model.generate_tag(image)

    return tags

def inference_ram_img_embed(image, model):

    with torch.no_grad():
        embedding = model.generate_image_embeds(image)

    return embedding

def inference_ram_openset(image, model):

    with torch.no_grad():
        tags = model.generate_tag_openset(image)

    return tags[0]

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, tensor_type, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embedding size must be divisible by number of heads."

        self.query_fc = nn.Linear(embed_size, embed_size)
        self.key_fc = nn.Linear(embed_size, embed_size)
        self.value_fc = nn.Linear(embed_size, embed_size)

        self.out_fc = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)

        # Scaled dot-product attention
        self.scale = self.head_dim**2


    def forward(self, query, key, value, mask=None):

        batch_size = query.size(0)
        # self.scale = self.scale.to(query.device)
        query = self.query_fc(query)  # (batch_size, query_len, embed_size)
        key = self.key_fc(key)  # (batch_size, key_len, embed_size)
        value = self.value_fc(value)  # (batch_size, value_len, embed_size)

        # Split the embedding into attention heads for parallel computation.
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)  # (batch_size, num_heads, query_len, head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)  # (batch_size, num_heads, key_len, head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)  # (batch_size, num_heads, value_len, head_dim)

        energy = torch.matmul(query, key.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy, dim=-1).to(value.dtype)  # (batch_size, num_heads, query_len, key_len)

        attention = self.dropout(attention)
        assert attention.dtype == value.dtype, 'Attention and value must have the same data type, but got {} and {}'.format(attention.dtype, value.dtype)
        out = torch.matmul(attention, value)  # (batch_size, num_heads, query_len, head_dim)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1,
                                                    self.num_heads * self.head_dim)  # (batch_size, query_len, embed_size)

        out = self.out_fc(out)  # (batch_size, query_len, embed_size)

        return out


class AttentionPoolingLayer(nn.Module):
    def __init__(self, args, input_dim, num_latent_queries, num_heads):
        super(AttentionPoolingLayer, self).__init__()
        if args.mixed_precision == "fp32":
            tensor_type = torch.float32
        else:
            tensor_type = torch.float16
        self.args = args
        self.num_latent_queries = num_latent_queries
        self.latent_queries = nn.Parameter(torch.randn(num_latent_queries, input_dim))
        self.multihead_attn = MultiHeadAttention(embed_size=input_dim, num_heads=num_heads, tensor_type=tensor_type)

    def forward(self, x):
        # x shape: (B, N, C)
        B, N, C = x.shape

        latent_queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1).to(x.device)
        # Multihead Attention

        attn_output = self.multihead_attn(latent_queries, x, x)

        return attn_output

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.silu = nn.SiLU()

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.silu(self.conv1(x))

        x = x.permute(0, 2, 1)
        return x


class AttentionPooling(AttentionPoolingLayer):
    def __init__(self, args, input_dim, num_latent_queries, num_heads=1):
        super(AttentionPooling, self).__init__(args, input_dim, num_latent_queries, num_heads)
        self.args = args
        self.embedding_change = Conv1x1(input_dim, input_dim*num_heads)
        self.attention_pool = AttentionPoolingLayer(args, input_dim*num_heads, num_latent_queries, num_heads)

    def forward(self, x):
        x = self.embedding_change(x)
        x = self.attention_pool(x)
        return x
