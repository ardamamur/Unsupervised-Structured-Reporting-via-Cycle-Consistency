import torch
from torch import nn
import torchvision.transforms as transforms
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder

class ModifiedMultiImageEncoder(nn.Module):
    def __init__(self, original_model):
        super(ModifiedMultiImageEncoder, self).__init__()
        self.original_model = original_model
        self.original_model.vit_pooler = self.original_model.vit_pooler.eval()

    def forward(self, x):
        # Extract features from ResNet layers
        ################### RESNET ####################################
        x = self.original_model.encoder.conv1(x)
        x = self.original_model.encoder.bn1(x)
        x = self.original_model.encoder.relu(x)
        x = self.original_model.encoder.maxpool(x)
    
        resnet_features = []
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.original_model.encoder, layer_name)
            x = layer(x)
            resnet_features.append(x)
    
        ###################### BACKBONE TO VIT ######################
        x = self.original_model.backbone_to_vit(x)
        x = x.flatten(2).transpose(1, 2)
        B, N, C = x.shape  # Adjusted to include channel dimension
    
        # Adjust positional and type embeddings
        pos_embed = self.original_model.vit_pooler.pos_embed[:, :N, :].repeat(B, 1, 1)
        type_embed = self.original_model.vit_pooler.type_embed[0].expand(B, N, -1)
    
        # Combine embeddings and pass through ViT pooler
        pos_and_type_embed = pos_embed + type_embed
        x = self.original_model.vit_pooler.pos_drop(x)
    
        vit_features = []
        for block in self.original_model.vit_pooler.blocks:
            x = block(x, pos_and_type_embed)
            vit_features.append(x)
    
        x = self.original_model.vit_pooler.norm_post(x)
    
        # Extract current patch features
    
        cur_img_token_id = 0
        current_token_features = x[:, cur_img_token_id : self.original_model.vit_pooler.num_patches + cur_img_token_id]
        
        # Calculate the spatial dimensions for each patch
        num_patches = self.original_model.vit_pooler.num_patches
        patch_height = patch_width = int(num_patches ** 0.5)
        
        # Reshape into the spatial grid of patches
        current_patch_features = current_token_features.view(B, patch_height, patch_width, -1).permute(0, 3, 1, 2)
        
        return resnet_features, vit_features
    
class perceptual_biovil_vit(nn.Module):
    def __init__(self, return_all=False):
        super(perceptual_biovil_vit, self).__init__()
        self.return_all = return_all
        self.original_model = get_biovil_t_image_encoder().encoder
        self.original_model = self.original_model.cuda()
        self.feature_extractor = ModifiedMultiImageEncoder(self.original_model)
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        true_resnet_features, true_vit_features = self.feature_extractor(y_true)
        pred_resnet_features, pred_vit_features = self.feature_extractor(y_pred)

        # Calculate loss for ResNet features
        resnet_loss = 0.0
        for true_feat, pred_feat in zip(true_resnet_features, pred_resnet_features):
            resnet_loss += nn.functional.mse_loss(true_feat, pred_feat)
        
        # Calculate loss for ViT features
        vit_loss = 0.0
        for true_feat, pred_feat in zip(true_vit_features, pred_vit_features):
            vit_loss += nn.functional.mse_loss(true_feat, pred_feat)

        # Combine the losses from ResNet and ViT features
        total_loss = resnet_loss + vit_loss
        if self.return_all:
            return total_loss, resnet_loss, vit_loss
        
        # else return average loss
        else:
            return total_loss