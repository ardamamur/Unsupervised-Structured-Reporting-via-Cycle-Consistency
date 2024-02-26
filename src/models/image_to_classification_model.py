import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
import health_multimodal.image
from health_multimodal.image.model.model import BaseImageModel
from health_multimodal.image.utils import ImageModelType
from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder, get_biovil_image_encoder
from torchmetrics import Accuracy, Precision, Recall, F1Score

class ArkModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, criterion, ark_pretrained_path):
        super(ArkModel, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224', num_classes=num_classes, pretrained=False)

        self.state_dict = torch.load(ark_pretrained_path, map_location="cpu")
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in self.state_dict:
                print(f"Removing key {k} from pretrained checkpoint")
                del self.state_dict[k]

        self.model.load_state_dict(self.state_dict, strict=False)
        self.lr = learning_rate
        self.criterion = criterion

    def forward(self, x):
        # Pass the input through the underlying model
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, labels = train_batch['target'], train_batch['report']
        output = self.forward(x)
        loss = self.criterion(output, labels)

        # Log training loss to Tensorboard
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch['target'], val_batch['report']
        output = self.forward(x)
        loss = self.criterion(output, labels)

        # Log training loss to Tensorboard
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, test_batch, batch_idx):
        x, labels = test_batch['target'], test_batch['report']
        output = self.forward(x)
        loss = self.criterion(output, labels)

        # Log training loss to Tensorboard
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss


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


class BioVILModel(pl.LightningModule):
    def __init__(self, embedding_size, num_classes, hidden_1, dropout_rate, criterion, opt):
        super(BioVILModel, self).__init__()

        self.vision_transformer = VisionTransformer()
        self.classification_head = self.ClassificationHead = ClassificationHead(input_size=embedding_size, num_classes=num_classes,
                                                     hidden_size_1=hidden_1,
                                                     dropout_rate=dropout_rate)
        self.lr = opt['report_generator']['learning_rate']
        self.criterion = criterion
        self.define_metrics()

    def forward(self, x):
        # Pass the input through the underlying model
        x = self.vision_transformer(x)
        x = self.classification_head(x)
        return x

    def define_metrics(self):
        self.val_metrics = {
            'accuracy_micro': Accuracy(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda:0'),
            'precision_micro': Precision(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda:0'),
            'recall_micro': Recall(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda:0'),
            'f1_micro': F1Score(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda:0'),
            'accuracy_macro': Accuracy(task="multilabel", average="macro", num_labels=self.num_classes).to('cuda:0'),
            'precision_macro': Precision(task="multilabel", average="macro", num_labels=self.num_classes).to('cuda:0'),
            'recall_macro': Recall(task="multilabel", average="macro", num_labels=self.num_classes).to('cuda:0'),
            'f1_macro': F1Score(task="multilabel", average="macro", num_labels=self.num_classes).to('cuda:0'),
            'overall_precision': [],
        }

        self.train_metrics = {
            'accuracy_micro': Accuracy(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda:0'),
            'precision_micro': Precision(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda:0'),
            'recall_micro': Recall(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda:0'),
            'f1_micro': F1Score(task="multilabel", average="micro", num_labels=self.num_classes).to('cuda:0'),
            'accuracy_macro': Accuracy(task="multilabel", average="macro", num_labels=self.num_classes).to('cuda:0'),
            'precision_macro': Precision(task="multilabel", average="macro", num_labels=self.num_classes).to('cuda:0'),
            'recall_macro': Recall(task="multilabel", average="macro", num_labels=self.num_classes).to('cuda:0'),
            'f1_macro': F1Score(task="multilabel", average="macro", num_labels=self.num_classes).to('cuda:0'),
            'overall_precision': []
        }

    def log_val_metrics(self, metrics, on_step):
        # log the metrics
        metrics = {f'{self.phase}_{key}': value for key, value in metrics.items()}
        self.log_dict(metrics, on_step=on_step, on_epoch=True, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return {optimizer, scheduler}

    def calculate_overall_precision(self, preds, targets, batch_nmb):
        exact_matches = torch.all(preds == targets, dim=1)
        true_positives = torch.sum(exact_matches).item()
        precision = true_positives / batch_nmb
        return precision

    def training_step(self, train_batch, batch_idx):
        x, labels = train_batch['target'], train_batch['report']
        batch_nmb = x.shape[0]
        output = self.forward(x)
        loss = self.criterion(output, labels)

        output_sigmoid = torch.sigmoid(output)
        output_0_1 = torch.where(output_sigmoid > 0.5, 1.0, 0.0)

        self.train_metrics['accuracy_micro'].update(output_0_1, labels)
        self.train_metrics['precision_micro'].update(output_0_1, labels)
        self.train_metrics['recall_micro'].update(output_0_1, labels)
        self.train_metrics['f1_micro'].update(output_0_1, labels)

        self.train_metrics['accuracy_macro'].update(output_0_1, labels)
        self.train_metrics['precision_macro'].update(output_0_1, labels)
        self.train_metrics['recall_macro'].update(output_0_1, labels)
        self.train_metrics['f1_macro'].update(output_0_1, labels)

        overall_precision = self.calculate_overall_precision(output_0_1, labels, batch_nmb)
        self.train_metrics['overall_precision'].append(overall_precision)

        # Log training loss to Tensorboard
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch['target'], val_batch['report']
        batch_nmb = x.shape[0]

        output = self.forward(x)
        loss = self.criterion(output, labels)

        output_sigmoid = torch.sigmoid(output)
        output_0_1 = torch.where(output_sigmoid > 0.5, 1.0, 0.0)

        self.val_metrics['accuracy_micro'].update(output_0_1, labels)
        self.val_metrics['precision_micro'].update(output_0_1, labels)
        self.val_metrics['recall_micro'].update(output_0_1, labels)
        self.val_metrics['f1_micro'].update(output_0_1, labels)

        self.val_metrics['accuracy_macro'].update(output_0_1, labels)
        self.val_metrics['precision_macro'].update(output_0_1, labels)
        self.val_metrics['recall_macro'].update(output_0_1, labels)
        self.val_metrics['f1_macro'].update(output_0_1, labels)

        overall_precision = self.calculate_overall_precision(output_0_1, labels, batch_nmb)
        self.val_metrics['overall_precision'].append(overall_precision)

        # Log training loss to Tensorboard
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        # log the metrics
        self.phase = 'train'
        train_log_metrics = {
                'accuracy_micro': self.train_metrics['accuracy_micro'].compute(),
                'precision_micro': self.train_metrics['precision_micro'].compute(),
                'recall_micro': self.train_metrics['recall_micro'].compute(),
                'f1_micro': self.train_metrics['f1_micro'].compute(),
                'accuracy_macro': self.train_metrics['accuracy_macro'].compute(),
                'precision_macro': self.train_metrics['precision_macro'].compute(),
                'recall_macro': self.train_metrics['recall_macro'].compute(),
                'f1_macro': self.train_metrics['f1_macro'].compute(),
                'overall_precision': torch.mean(torch.tensor(self.train_metrics['overall_precision']))
        }

        self.log_val_metrics(train_log_metrics, on_step=False)

        # reset the metrics
        self.train_metrics['accuracy_micro'].reset()
        self.train_metrics['precision_micro'].reset()
        self.train_metrics['recall_micro'].reset()
        self.train_metrics['f1_micro'].reset()
        self.train_metrics['accuracy_macro'].reset()
        self.train_metrics['precision_macro'].reset()
        self.train_metrics['recall_macro'].reset()
        self.train_metrics['f1_macro'].reset()
        self.train_metrics['overall_precision'] = []

    def on_validation_epoch_end(self):
        self.phase = 'val'
        # log the metrics
        val_log_metrics = {
                'accuracy_micro': self.val_metrics['accuracy_micro'].compute(),
                'precision_micro': self.val_metrics['precision_micro'].compute(),
                'recall_micro': self.val_metrics['recall_micro'].compute(),
                'f1_micro': self.val_metrics['f1_micro'].compute(),
                'accuracy_macro': self.val_metrics['accuracy_macro'].compute(),
                'precision_macro': self.val_metrics['precision_macro'].compute(),
                'recall_macro': self.val_metrics['recall_macro'].compute(),
                'f1_macro': self.val_metrics['f1_macro'].compute(),
                'overall_precision': torch.mean(torch.tensor(self.val_metrics['overall_precision']))

        }
        self.log_val_metrics(val_log_metrics, on_step=False)

        # reset the metrics
        self.val_metrics['accuracy_micro'].reset()
        self.val_metrics['precision_micro'].reset()
        self.val_metrics['recall_micro'].reset()
        self.val_metrics['f1_micro'].reset()
        self.val_metrics['accuracy_macro'].reset()
        self.val_metrics['precision_macro'].reset()
        self.val_metrics['recall_macro'].reset()
        self.val_metrics['f1_macro'].reset()
        self.val_metrics['overall_precision'] = []

    def test_step(self, test_batch, batch_idx):
        pass

