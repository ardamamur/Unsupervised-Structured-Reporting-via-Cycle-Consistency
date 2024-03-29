import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
import health_multimodal.image
from health_multimodal.image.model.model import BaseImageModel
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.model.modules import MultiTaskModel
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        # self.image_inference = health_multimodal.image.get_image_inference(ImageModelType.BIOVIL_T)
        self.model = get_biovil_t_image_encoder()
        # print(self.model)

    def forward(self, x):
        x = self.model.forward(x).img_embedding
        # Check if normalization is needed
        x = torch.nn.functional.normalize(x, dim=1)
        return x



class ClassificationHead(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size_1, dropout_rate):
        super(ClassificationHead, self).__init__()
        hidden_dim_1 = hidden_size_1
        dropout_prob = dropout_rate
        self.fc1 = nn.Linear(input_size, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class BioViL(nn.Module):
    def __init__(self, embedding_size, num_classes, hidden_1, dropout_rate):
        super(BioViL, self).__init__()
        self.VisionTransformer = VisionTransformer()
        self.ClassificationHead = ClassificationHead(input_size=embedding_size, num_classes=num_classes,
                                                     hidden_size_1=hidden_1,
                                                     dropout_rate=dropout_rate)

    def forward(self, x):
        x = self.VisionTransformer(x)
        x = self.ClassificationHead(x)
        return x


class BioViL_V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_model = get_biovil_t_image_encoder()
        self.multi_task_classifier = MultiTaskModel(
            input_dim=512, 
            classifier_hidden_dim=256, 
            num_classes=1, 
            num_tasks=13  # Assuming num_classes = num_tasks for Chexpert
        )

    def forward(self, x):
        # Pass input through the image model, which includes the downstream classifier
        x = self.image_model(x).img_embedding
        x = self.multi_task_classifier(x)
        x = x.squeeze(1)  # Specify dimension to squeeze
        return x