
import torch
import torch.nn as nn
#import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F


class ViewLayer(nn.Module):
    def __init__(self, channels, height, width):
        super(ViewLayer, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)


class cGAN(nn.Module):
    def __init__(self, generator_layer_size, z_size, img_size, class_num):
        super().__init__()

        self.z_size = z_size
        self.img_size = img_size

        # self.label_emb = nn.Embedding(class_num, class_num)

        self.model = nn.Sequential(
            nn.Linear(self.z_size + class_num, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], self.img_size * self.img_size * 3),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Reshape z
        z = z.view(-1, self.z_size)
        # One-hot vector to embedding vector
        # c = self.label_emb(labels)
        # Concat image & label
        c = labels
        print(f'c_shape_gen:{c.shape}')
        print(f'z_shape_gen:{z.shape}')
        x = torch.cat([z, c], 1)
        print(f'x_shape_gen:{x.shape}')
        # Generator out
        out = self.model(x)

        return out.view(-1, 3, self.img_size, self.img_size)


class cGANconv(nn.Module):
    def __init__(self, z_size, img_size, class_num, img_channels):
        super(cGANconv, self).__init__()
        self.z_size = z_size
        self.features = 64
        self.img_size = img_size
        self.label_dim = 1
        self.img_channels = img_channels

        self.model = nn.Sequential(
            # Linear Layer as input
            nn.Linear(self.z_size + class_num, 128 * (self.img_size // 4) * self.img_size // 4),
            # Reshape to starting image dimensions (e.g., 128 x (img_size/4) x (img_size/4))
            ViewLayer(128, self.img_size // 4, self.img_size // 4),

            # Up-sampling layers
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer - adjust to match the number of output channels
            nn.Conv2d(64, self.img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Use Tanh for normalizing the output to [-1, 1]
        )

    def forward(self, z, labels):
        # Reshape z
        z = z.view(-1, self.z_size)

        c = labels
        x = torch.cat([z, c], 1)
        out = self.model(x)

        return out.view(-1, self.img_channels, self.img_size, self.img_size)


class cGANconv_v1(nn.Module):
    def __init__(self, z_size, img_size, class_num, img_channels):
        super(cGANconv_v1, self).__init__()
        self.z_size = z_size
        self.features = 64
        self.img_size = img_size
        self.label_dim = 1
        self.img_channels = img_channels

        self.model = nn.Sequential(
            # Linear Layer as input
            nn.Linear(self.z_size + class_num, 128 * (self.img_size // 4) * self.img_size // 4),
            # Reshape to starting image dimensions (e.g., 128 x (img_size/4) x (img_size/4))
            ViewLayer(128, self.img_size // 4, self.img_size // 4),

            # Up-sampling layers
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Output layer - adjust to match the number of output channels
            nn.Conv2d(64, self.img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Use Tanh for normalizing the output to [-1, 1]
        )

    def forward(self, z, labels):
        # Reshape z
        z = z.view(-1, self.z_size)

        c = labels
        x = torch.cat([z, c], 1)
        out = self.model(x)

        return out.view(-1, self.img_channels, self.img_size, self.img_size)


class cGANconv_V2(nn.Module):
    def __init__(self, z_size, img_size, class_num, img_channels, projection_dim=128):
        super(cGANconv_V2, self).__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.z_size = z_size
        self.class_num = class_num
        self.projection_dim = projection_dim
        self.feature_map_size = 256

        self.disease_projection = nn.Sequential(
            nn.Linear(self.class_num, self.projection_dim),
            nn.LeakyReLU(0.2)
        )

        self.initial_layer = nn.Sequential(
            nn.Linear(self.z_size + self.projection_dim, 128 * (self.img_size // 4) * (self.img_size // 4)),
            ViewLayer(128, self.img_size // 4, self.img_size // 4),
            nn.ReLU(0.2)
        )

        self.upsample_layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(124, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )

        self.output_layer = nn.Sequential(
            nn.Conv2d(64, self.img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.adaptive_layer = nn.Linear(projection_dim, self.feature_map_size)


    def forward(self, z, labels):
        disease_projection = self.disease_projection(labels)
        z_code = z.view(-1, self.z_size)
        combined_input = torch.cat([z_code, disease_projection], dim=1)

        x = self.initial_layer(combined_input)
        x = self.upsample_layers(x)
        img = self.output_layer(x)

        return img.view(-1, self.img_channels, self.img_size, self.img_size)


class ViewLayer_v1(nn.Module):
    def __init__(self, size):
        super(ViewLayer_v1, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(*self.size)


class cGANconv_v3(nn.Module):
    def __init__(self, z_size, img_size, class_num, img_channels):
        super(cGANconv_v3, self).__init__()
        self.z_size = z_size
        self.img_size = img_size
        self.class_num = class_num
        self.img_channels = img_channels
        self.label_embedding_dim = 50  # Dimension for the embedded labels

        # Label embedding layer
        self.label_embedding = nn.Sequential(
            nn.Linear(class_num, self.label_embedding_dim),
            nn.LeakyReLU(0.2)
        )

        self.model = nn.Sequential(
            # Linear Layer as input, incorporating embedded label dimension
            nn.Linear(self.z_size + self.label_embedding_dim, 128 * (self.img_size // 4) * (self.img_size // 4)),
            ViewLayer_v1((-1, 128, self.img_size // 4, self.img_size // 4)),

            # Up-sampling layers with conditional inputs
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Injecting label information again
            ViewLayer_v1((-1, 128 + self.label_embedding_dim, self.img_size // 2, self.img_size // 2)),
            nn.Conv2d(128 + self.label_embedding_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Final up-sampling and output layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, self.img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Use Tanh for normalizing the output to [-1, 1]
        )

        print(self.model)

    def forward(self, z, labels):
        # Embedding the labels
        embedded_labels = self.label_embedding(labels)

        # Initial concatenation of noise vector and embedded labels
        z = z.view(-1, self.z_size)
        x = torch.cat([z, embedded_labels], 1)
        out = self.model[0:6](x)  # Process through the first part of the model

        # Concatenate embedded labels again with the intermediate output
        embedded_labels = embedded_labels.unsqueeze(2).unsqueeze(3)
        embedded_labels = embedded_labels.expand(-1, -1, self.img_size // 2, self.img_size // 2)
        out = torch.cat([out, embedded_labels], 1)

        # Process through the final part of the model
        out = self.model[6:](out)

        return out.view(-1, self.img_channels, self.img_size, self.img_size)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        out = self.relu(out)
        return out


class SelfAttention(nn.Module):
    """ Self-attention layer for cGANconv model """
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        return out + x  # Skip connection


class cGANconv_v4(nn.Module):
    def __init__(self, z_size, img_size, class_num, img_channels):
        super(cGANconv_v4, self).__init__()
        self.z_size = z_size
        self.img_size = img_size
        self.class_num = class_num
        self.img_channels = img_channels
        self.label_embedding_dim = 50  # Dimension for the embedded labels

        # Label embedding layer
        self.label_embedding = nn.Sequential(
            nn.Linear(class_num, self.label_embedding_dim),
            nn.LeakyReLU(0.2)
        )

        self.model = nn.Sequential(
            # Linear Layer as input, incorporating embedded label dimension
            nn.Linear(self.z_size + self.label_embedding_dim, 128 * (self.img_size // 4) * (self.img_size // 4)),
            ViewLayer_v1((-1, 128, self.img_size // 4, self.img_size // 4)),

            # Up-sampling layers with conditional inputs
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            SelfAttention(128),  # Attention layer after first convolution
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            SelfAttention(64),  # Second attention layer
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Final up-sampling and output layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, self.img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Use Tanh for normalizing the output to [-1, 1]
        )

        print(self.model)

    def forward(self, z, labels):
        # Embedding the labels
        embedded_labels = self.label_embedding(labels)

        # Initial concatenation of noise vector and embedded labels
        z = z.view(-1, self.z_size)
        x = torch.cat([z, embedded_labels], 1)
        out = self.model(x)

        return out.view(-1, self.img_channels, self.img_size, self.img_size)


class cGANconv_v5(nn.Module):
    def __init__(self, z_size, img_size, class_num, img_channels):
        super(cGANconv_v5, self).__init__()
        self.z_size = z_size
        self.img_size = img_size
        self.class_num = class_num
        self.img_channels = img_channels
        self.label_embedding_dim = 50  # Dimension for the embedded labels

        # Label embedding layer
        self.label_embedding = nn.Sequential(
            nn.Linear(class_num, self.label_embedding_dim),
            nn.LeakyReLU(0.2)
        )

        self.model = nn.Sequential(
            # Linear Layer as input, incorporating embedded label dimension
            nn.Linear(self.z_size + self.label_embedding_dim, 128 * (self.img_size // 4) * (self.img_size // 4)),
            ViewLayer_v1((-1, 128, self.img_size // 4, self.img_size // 4)),

            # Up-sampling layers with conditional inputs
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            SelfAttention(128),  # Attention layer after first convolution
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            ViewLayer_v1((-1, 128 + self.label_embedding_dim, self.img_size // 2, self.img_size // 2)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            SelfAttention(64),  # Second attention layer
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Final up-sampling and output layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, self.img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Use Tanh for normalizing the output to [-1, 1]
        )

        print(self.model)

    def forward(self, z, labels):
        # Embedding the labels
        embedded_labels = self.label_embedding(labels)

        # Initial concatenation of noise vector and embedded labels
        z = z.view(-1, self.z_size)
        x = torch.cat([z, embedded_labels], 1)
        out = self.model[0:7](x)  # Process through the first part of the model

        # Concatenate embedded labels again with the intermediate output
        embedded_labels = embedded_labels.unsqueeze(2).unsqueeze(3)
        embedded_labels = embedded_labels.expand(-1, -1, self.img_size // 2, self.img_size // 2)
        out = torch.cat([out, embedded_labels], 1)

        # Process through the final part of the model
        out = self.model[7:](out)

        return out.view(-1, self.img_channels, self.img_size, self.img_size)


class cGANconv_v6(nn.Module):
    def __init__(self, z_size, img_size, class_num, img_channels):
        super(cGANconv_v6, self).__init__()
        self.z_size = z_size
        self.img_size = img_size
        self.class_num = class_num
        self.img_channels = img_channels
        self.label_embedding_dim = 50  # Dimension for the embedded labels

        # Label embedding layer
        self.label_embedding = nn.Sequential(
            nn.Linear(class_num, self.label_embedding_dim),
            nn.LeakyReLU(0.2)
        )

        self.model = nn.Sequential(
            # Linear Layer as input, incorporating embedded label dimension
            nn.Linear(self.z_size + self.label_embedding_dim, 128 * (self.img_size // 4) * (self.img_size // 4)),
            ViewLayer_v1((-1, 128, self.img_size // 4, self.img_size // 4)),

            # Up-sampling layers with conditional inputs
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            SelfAttention(128),  # Attention layer after first convolution
            ResNetBlock(128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            SelfAttention(64),  # Second attention layer
            ResNetBlock(64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Final up-sampling and output layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, self.img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Use Tanh for normalizing the output to [-1, 1]
        )

        print(self.model)

    def forward(self, z, labels):
        # Embedding the labels
        embedded_labels = self.label_embedding(labels)

        # Initial concatenation of noise vector and embedded labels
        z = z.view(-1, self.z_size)
        x = torch.cat([z, embedded_labels], 1)
        out = self.model(x)

        return out.view(-1, self.img_channels, self.img_size, self.img_size)


class cGANconv_v7(nn.Module):
    def __init__(self, z_size, img_size, class_num, img_channels):
        super(cGANconv_v7, self).__init__()
        self.z_size = z_size
        self.img_size = img_size
        self.class_num = class_num
        self.img_channels = img_channels
        self.label_embedding_dim = 50  # Dimension for the embedded labels

        # Label embedding layer
        self.label_embedding = nn.Sequential(
            nn.Linear(class_num, self.label_embedding_dim),
            nn.LeakyReLU(0.2)
        )

        self.model = nn.Sequential(
            # Linear Layer as input, incorporating embedded label dimension
            nn.Linear(self.z_size + self.label_embedding_dim, 128 * (self.img_size // 4) * (self.img_size // 4)),
            ViewLayer_v1((-1, 128, self.img_size // 4, self.img_size // 4)),

            # Up-sampling layers with conditional inputs
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            SelfAttention(128),  # Attention layer after first convolution
            ResNetBlock(128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            ViewLayer_v1((-1, 128 + self.label_embedding_dim, self.img_size // 2, self.img_size // 2)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            SelfAttention(64),  # Second attention layer
            ResNetBlock(64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Final up-sampling and output layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, self.img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Use Tanh for normalizing the output to [-1, 1]
        )

        print(self.model)

    def forward(self, z, labels):
        # Embedding the labels
        embedded_labels = self.label_embedding(labels)

        # Initial concatenation of noise vector and embedded labels
        z = z.view(-1, self.z_size)
        x = torch.cat([z, embedded_labels], 1)
        out = self.model[0:8](x)  # Process through the first part of the model

        # Concatenate embedded labels again with the intermediate output
        embedded_labels = embedded_labels.unsqueeze(2).unsqueeze(3)
        embedded_labels = embedded_labels.expand(-1, -1, self.img_size // 2, self.img_size // 2)
        out = torch.cat([out, embedded_labels], 1)

        # Process through the final part of the model
        out = self.model[8:](out)

        return out.view(-1, self.img_channels, self.img_size, self.img_size)


class CrossAttention(nn.Module):
    def __init__(self, feature_dim, label_dim, img_size):
        super(CrossAttention, self).__init__()
        self.img_size = img_size
        self.feature_conv = nn.Conv2d(feature_dim, feature_dim // 8, 1)
        self.label_conv = nn.Conv2d(label_dim, feature_dim // 8, 1)
        self.value_conv = nn.Conv2d(feature_dim, feature_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature_maps, labels):
        batch, channels, height, width = feature_maps.size()
        label_query = self.label_conv(labels).view(batch, -1, height * width)
        feature_key = self.feature_conv(feature_maps).view(batch, -1, height * width).permute(0, 2, 1)
        value = self.value_conv(feature_maps).view(batch, -1, height * width)

        attention = self.softmax(torch.bmm(label_query, feature_key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)

        return out + feature_maps  # Skip connection

class cGANconv_v8(nn.Module):
    def __init__(self, z_size, img_size, class_num, img_channels):
        super(cGANconv_v8, self).__init__()
        self.z_size = z_size
        self.img_size = img_size
        self.class_num = class_num
        self.img_channels = img_channels
        self.label_embedding_dim = 50  # Dimension for the embedded labels

        # Label embedding layer
        self.label_embedding = nn.Sequential(
        nn.Linear(class_num, self.label_embedding_dim),
        nn.LeakyReLU(0.2)
        )

        # Define Cross-Attention layer
        self.cross_attention1 = CrossAttention(128, self.label_embedding_dim, self.img_size // 2)
        self.cross_attention2 = CrossAttention(64, self.label_embedding_dim, self.img_size)

        # Model layers
        self.model = nn.Sequential(
            # Linear Layer as input, incorporating embedded label dimension
            nn.Linear(self.z_size + self.label_embedding_dim, 128 * (self.img_size // 4) * (self.img_size // 4)),
            ViewLayer_v1((-1, 128, self.img_size // 4, self.img_size // 4)),

            # Up-sampling layers with conditional inputs
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            SelfAttention(128),  # Attention layer after first convolution
            # Inject Cross-Attention here
            ResNetBlock(128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            ViewLayer_v1((-1, 128 + self.label_embedding_dim, self.img_size // 2, self.img_size // 2)),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            SelfAttention(64),  # Second attention layer
            # Inject Cross Attention here
            ResNetBlock(64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Final up-sampling and output layer
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, self.img_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Use Tanh for normalizing the output to [-1, 1]
        )
        print(self.model)

    def forward(self, z, labels):
        # Embedding the labels
        embedded_labels = self.label_embedding(labels)

        # Initial concatenation of noise vector and embedded labels
        z = z.view(-1, self.z_size)
        x = torch.cat([z, embedded_labels], 1)
        out = self.model[0:5](x)  # Adjust indices as needed

        # Apply first cross-attention
        embedded_labels_expanded = embedded_labels.unsqueeze(2).unsqueeze(3)
        embedded_labels_expanded = embedded_labels_expanded.expand(-1, -1, self.img_size // 2,
                                                                           self.img_size // 2)
        out = self.cross_attention1(out, embedded_labels_expanded)

        # Continue through the model
        out = self.model[5:11](out)

        # Apply second cross-attention
        embedded_labels_expanded = embedded_labels_expanded.expand(-1, -1, self.img_size, self.img_size)
        out = self.cross_attention2(out, embedded_labels_expanded)

        # Final layers
        out = self.model[11:](out)

        return out.view(-1, self.img_channels, self.img_size, self.img_size)


class Discriminator(nn.Module):
    def __init__(self, discriminator_layer_size, img_size, class_num):
        super().__init__()

        # self.label_emb = nn.Embedding(class_num, class_num)
        self.img_size = img_size

        self.model = nn.Sequential(
            nn.Linear(3 * self.img_size * self.img_size + class_num, discriminator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[0], discriminator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[1], discriminator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[2], 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # Reshape fake image
        # print(f'x_shape: {x.shape}')
        x = x.view(-1, 3 * self.img_size * self.img_size)
        # print(x.shape)
        # One-hot vector to embedding vector
        # c = self.label_emb(labels)
        c = labels
        print(f'x_shape_disc:{x.shape}')
        print(f'c_shape_disc: {c.shape}')
        # Concat image & label
        x = torch.cat([x, c], 1)
        print(f'x_shape_disc: {x.shape}')
        # Discriminator out
        out = self.model(x)

        return out.squeeze()

if __name__ == '__main__':
    print("model_v3")
    test1 = cGANconv_v3(100, 224, 13, 3)
    print("model_v4")
    test2 = cGANconv_v4(100, 224, 13, 3)
    print("model_v5")
    test3 = cGANconv_v5(100, 224, 13, 3)
    print("model_v6")
    test4 = cGANconv_v6(100, 224, 13, 3)
    print("model_v7")
    test5 = cGANconv_v7(100, 224, 13, 3)
    print("model_v8")
    test6 = cGANconv_v8(100, 224, 13, 3)


