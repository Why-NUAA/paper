import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.net(x)
        x = self.norm(x + residual)
        return x


class MyAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super(MyAttention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_Q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_K = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_V = nn.Linear(dim, dim_head * heads, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        residual = x
        b, n_x, d_x, h_x = *x.shape, self.heads
        b, n_y, d_y, h_y = *y.shape, self.heads

        q = self.to_Q(x).view(b, -1, self.heads, d_y // self.heads).transpose(1, 2)
        k = self.to_K(y).view(b, -1, self.heads, d_x // self.heads).transpose(1, 2)
        v = self.to_V(y).view(b, -1, self.heads, d_x // self.heads).transpose(1, 2)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out + residual)
        return out


class MyCrossAttention(nn.Module):
    def __init__(self, rois, dim, heads, dim_head, dropout):
        super(MyCrossAttention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_Q = nn.Linear(rois, rois, bias=False)
        self.to_K = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_V = nn.Linear(dim, dim_head * heads, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        residual = y
        b, n_x, d_x, h_x = *x.shape, self.heads
        b, n_y, d_y, h_y = *y.shape, self.heads

        q = self.to_Q(x).view(b, -1, 1, d_x).transpose(1, 2)
        q = q.repeat(1, self.heads, 1, 1)
        k = self.to_K(y).view(b, -1, self.heads, d_y // self.heads).transpose(1, 2)
        v = self.to_V(y).view(b, -1, self.heads, d_y // self.heads).transpose(1, 2)

        kkt = einsum('b h i d, b h j d -> b h i j', k, k) * self.scale
        dots = einsum('b h i d, b h j d -> b h i j', q, kkt) * self.scale
        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.norm(out + residual)
        return out


class MyDecoderLayer(nn.Module):
    def __init__(self, rois, dim, heads, dim_head, mlp_dim, dropout=0.5):
        super(MyDecoderLayer, self).__init__()
        self.SelfAttention = MyAttention(rois, heads=1, dim_head=rois)
        self.CrossAttention = MyCrossAttention(rois, dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.FeedForward = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, enc_out):
        out = self.SelfAttention(x, x)
        out = self.CrossAttention(out, enc_out)
        out = self.FeedForward(out)
        return out


class MyDecoder(nn.Module):
    def __init__(self, rois, dim, depth, heads, dim_head, mlp_dim, dropout):
        super(MyDecoder, self).__init__()
        self.layers = nn.ModuleList(
            [MyDecoderLayer(rois, dim, heads, dim_head, mlp_dim, dropout=dropout) for _ in range(depth)])

    def forward(self, x, enc_out):
        for layer in self.layers:
            x = layer(x, enc_out)
        return x


class MyEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout):
        super(MyEncoderLayer, self).__init__()
        self.SelfAttention = MyAttention(dim, heads, dim_head)
        self.norm = nn.LayerNorm(dim)
        self.FeedForward = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = self.SelfAttention(x, x)
        x = self.FeedForward(x)
        return x


class MyEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super(MyEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [MyEncoderLayer(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class net(nn.Module):
    def __init__(self, rois, timepoints, num_classes, depth, heads, dropout):
        super(net, self).__init__()
        self.dim = timepoints
        self.rois = rois
        mlp_dim = self.dim * 3

        self.encoder = MyEncoder(self.dim, depth, heads, self.dim // heads, mlp_dim, dropout)
        self.decoder = MyDecoder(rois, self.dim, depth, heads, self.dim // heads, mlp_dim, dropout)

        self.to_latent = nn.Identity()
        self.fc1 = nn.Sequential(
            nn.Linear(45 * 45, 1024),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes)
        )
        self.norm = nn.LayerNorm(self.rois)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=(2, 2), padding=(0, 0))

    def forward(self, inputs):
        mri = inputs[:, :, : -self.rois]
        dti = inputs[:, :, -self.rois:]

        x = self.encoder(mri)
        x_out = torch.matmul(x, x.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.rois))

        y = self.decoder(dti, x)
        y_out = torch.matmul(y, y.transpose(-1, -2) / torch.sqrt(torch.tensor(self.rois)))

        out = x_out + y_out
        out = self.maxpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)

        out_norm = F.normalize(out, p=2, dim=1)
        return out_norm, out

    def frozen_forward(self, x):
        with torch.no_grad():
            _, x = self.forward(x)
        x = self.mlp_head(x)
        return torch.softmax(x, dim=-1)