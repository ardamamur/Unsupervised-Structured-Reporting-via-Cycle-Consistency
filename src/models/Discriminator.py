import torch.nn as nn
import torch

class ReportDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super(ReportDiscriminator, self).__init__()

        self.output_shape = (1, )

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),  # Increased size
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Added dropout for regularization
            
            nn.Linear(512, 256),  # Additional layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Additional dropout
            
            nn.Linear(256, 128),  # Additional layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),  # Additional dropout
            
            nn.Linear(128, 1),  # Output layer
        )

    def forward(self, x):
        # Removed the sigmoid activation before the model as it's not usual to pre-activate inputs
        x = self.model(x)
        x = torch.sigmoid(x)  # Sigmoid at the output for binary classification
        return x



class ImageDiscriminator(nn.Module):
    def __init__(self, input_shape):
        super(ImageDiscriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_channels, out_channels, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # C64 -> C128 -> C256 -> C512
        self.model = nn.Sequential(
            *discriminator_block(channels, out_channels=64, normalize=False),
            *discriminator_block(64, out_channels=128),
            *discriminator_block(128, out_channels=256),
            *discriminator_block(256, out_channels=512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, padding=1)
        )

    def forward(self, img):
        x = self.model(img)
        x = torch.sigmoid(x)
        return x
