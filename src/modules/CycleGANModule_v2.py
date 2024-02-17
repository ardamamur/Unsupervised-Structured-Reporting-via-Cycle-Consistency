from typing import Union, List
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import deepspeed as ds
import os
import io
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from models.buffer import ReportBuffer, ImageBuffer
from utils.get_models import load_networks
from losses.vgg import perceptual_vgg
from losses.biovil import perceptual_biovil
from losses.biovil_t import perceptual_biovil_vit
from losses.ark_v1 import perceptual_ark_v1
from losses.ark_v2 import perceptual_ark_v2
from torchmetrics import Accuracy, Precision, Recall, F1Score
from utils.environment_settings import env_settings
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
from torch.autograd import Variable
torch.autograd.set_detect_anomaly(True)

class CycleGAN(pl.LightningModule):
    """
    Generator for image-to-report translation (report_generator).
    Generator for report-to-image translation (image_generator).
    Discriminator for Generated Reports (report_dicriminator). --> Currenlty using ClassificationLoss (Cosine Similarity)
    Discriminator for Generated Images (image_generator).    
    """

    def __init__(self, opt):
        super(CycleGAN, self).__init__()
        self.save_hyperparameters()
        self.initialize_variables()
        self.initialize_models()
        self.initialize_losses()
        self.initialize_metrics()
        self.initialize_optimizers()

    def initialize_variables(self):
        self.data_imputation = opt['dataset']['data_imputation']
        self.input_size = opt["dataset"]["input_size"]
        self.num_classes = opt['dataset']['num_classes']
        self.batch_size = opt['dataset']['batch_size']
        self.n_epochs = opt['trainer']['n_epoch']
        self.z_size = opt['image_generator']['z_size']
        self.save_images = opt['trainer']['save_images']
        self.lambda_cycle = opt['trainer']['lambda_cycle']
        self.buffer_size = opt['trainer']['buffer_size']
        self.log_images_freq = opt['trainer']['log_images_steps']
        if self.save_images:
            self.sample_dir = env_settings.SAVE_IMAGES_PATH
            os.makedirs(self.sample_dir, exist_ok=True)
            print(f"Images will be saved at {self.sample_dir}")

    def initialize_models(self):
        self.report_generator, self.report_discriminator, self.image_generator, self.image_discriminator = load_networks(self.opt)
        self.buffer_reports = ReportBuffer(self.buffer_size)
        self.buffer_images = ImageBuffer(self.buffer_size)
        self.gen_threshold = self.opt["trainer"]["gen_threshold"]
        self.disc_threshold = self.opt["trainer"]["disc_threshold"]

    def initialize_optimizers(self):
        optimizer_dict = {
            'Adam': ds.ops.adam.FusedAdam,
            'AdamW': torch.optim.AdamW,
        }
        self.image_gen_optimizer = optimizer_dict[opt["image_generator"]["optimizer"]]
        self.image_disc_optimizer = optimizer_dict[opt["image_discriminator"]["optimizer"]]
        self.report_gen_optimizer = optimizer_dict[opt["report_generator"]["optimizer"]]
        self.report_disc_optimizer = optimizer_dict[opt["report_discriminator"]["optimizer"]]

    def initialize_losses(self):
        self.img_consistency_loss = perceptual_biovil_bit()
        self.report_consistency_loss = nn.BCEWithLogitsLoss()
        self.report_adversarial_loss = nn.MSELoss()
        self.image_adversarial_loss = nn.MSELoss()
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

    def initialize_metrics(self):
        self.macro_accuracy = {
            'train': Accuracy(task="multilabel", average="macro", num_labels=self.num_classes),
            'val': Accuracy(task="multilabel", average="macro", num_labels=self.num_classes),
        }
        self.micro_accuracy = {
            'train': Accuracy(task="multilabel", average="micro", num_labels=self.num_classes),
            'val': Accuracy(task="multilabel", average="micro", num_labels=self.num_classes),
        }
        self.macro_f1 = {
            'train': F1Score(task="multilabel", average="macro", num_labels=self.num_classes),
            'val': F1Score(task="multilabel", average="macro", num_labels=self.num_classes),
        }
        self.micro_f1 = {
            'train': F1Score(task="multilabel", average="micro", num_labels=self.num_classes),
            'val': F1Score(task="multilabel", average="micro", num_labels=self.num_classes),
        }
        self.macro_precision = {
            'train': Precision(task="multilabel", average="macro", num_labels=self.num_classes),
            'val': Precision(task="multilabel", average="macro", num_labels=self.num_classes),
        }
        self.micro_precision = {
            'train': Precision(task="multilabel", average="micro", num_labels=self.num_classes),
            'val': Precision(task="multilabel", average="micro", num_labels=self.num_classes),
        }
        self.macro_recall = {
            'train': Recall(task="multilabel", average="macro", num_labels=self.num_classes),
            'val': Recall(task="multilabel", average="macro", num_labels=self.num_classes),
        }
        self.micro_recall = {
            'train': Recall(task="multilabel", average="micro", num_labels=self.num_classes),
            'val': Recall(task="multilabel", average="micro", num_labels=self.num_classes),
        }

    def forward(self, x):
        x = x.float().to(self.device)
        y = self.report_generator(x)
        y = torch.sigmoid(y)
        y = torch.where(y > 0.5, 1, 0)
        return y
 
    def get_lr_scheduler(self, optimizer, decay_epochs):
        def lr_lambda(epoch):
            len_decay_phase = self.n_epochs - decay_epochs + 1.0
            curr_decay_step = max(0, epoch - decay_epochs + 1.0)
            val = 1.0 - curr_decay_step / len_decay_phase
            return max(0.0, val)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def configure_optimizers(self):
        def configure_optimizers(self):
        image_gen_opt_config = {
            "lr" : self.opt["image_generator"]["learning_rate"],
            "betas" : (self.opt["image_generator"]["beta1"], self.opt["image_generator"]["beta2"])
        }
        image_generator_optimizer = self.image_gen_optimizer(
            list(self.image_generator.parameters()),
            **image_gen_opt_config,
        )
        image_generator_scheduler = self.get_lr_scheduler(image_generator_optimizer, self.opt["image_generator"]["decay_epochs"])


        report_gen_opt_config = {
            "lr" : self.opt["report_generator"]["learning_rate"],
            "betas" : (self.opt["report_generator"]["beta1"], self.opt["report_generator"]["beta2"])
        }
        report_generator_optimizer = self.report_gen_optimizer(
            list(self.report_generator.parameters()),
            **report_gen_opt_config,
        )
        report_generator_scheduler = self.get_lr_scheduler(report_generator_optimizer, self.opt["report_generator"]["decay_epochs"])


        image_disc_opt_config = {
            "lr" : self.opt["image_discriminator"]["learning_rate"],
            "betas" : (self.opt["image_discriminator"]["beta1"], self.opt["image_discriminator"]["beta2"])
        }
        image_discriminator_optimizer = self.image_disc_optimizer(
            list(self.image_discriminator.parameters()),
            **image_disc_opt_config,
        )
        image_discriminator_scheduler = self.get_lr_scheduler(image_discriminator_optimizer, self.opt["image_discriminator"]["decay_epochs"])

        report_disc_opt_config = {
            "lr" : self.opt["report_discriminator"]["learning_rate"],
            "betas" : (self.opt["report_discriminator"]["beta1"], self.opt["report_discriminator"]["beta2"])
        }
        report_discriminator_optimizer = self.report_disc_optimizer(
            list(self.report_discriminator.parameters()),
            **report_disc_opt_config,
        )
        report_discriminator_scheduler = self.get_lr_scheduler(report_discriminator_optimizer, self.opt["report_discriminator"]["decay_epochs"])
        optimizers = [image_generator_optimizer, report_generator_optimizer image_discriminator_optimizer, report_discriminator_optimizer]
        schedulers = [image_generator_scheduler, report_generator_scheduler, image_discriminator_scheduler, report_discriminator_scheduler]


        return optimizers, schedulers


    def img_adv_criterion(self, fake_image, real_image):
        # adversarial loss
        return self.img_adversarial_loss(fake_image, real_image)

    def img_consistency_criterion(self, real_image, cycle_image):
        # reconstruction loss
        return self.img_consistency_loss(real_image, cycle_image)
    
    def report_consistency_criterion(self, real_report, cycle_report):
        # reconstruction loss
        return self.report_consistency_loss(real_report, cycle_report)

    def report_adv_criterion(self, fake_report, real_report):
        # adversarial loss
        return self.report_adversarial_loss(fake_report, real_report)

    def generator_step(self, valid_img, valid_report, mode="train"):
        # calculate loss for generator
        # adversarial loss
        adv_loss_IR = self.report_adv_criterion(self.report_discriminator(self.fake_report), valid_report)
        adv_loss_RI = self.img_adv_criterion(self.image_discriminator(self.fake_img), valid_img)
        total_adv_loss = adv_loss_IR + adv_loss_RI
        ############################################################################################
        # cycle loss
        cycle_loss_IRI_perceptul = self.img_consistency_criterion(self.real_img, self.cycle_img)
        cycle_loss_IRI_MSE = self.MSE(self.real_img, self.cycle_img)
        cycle_loss_RIR = self.report_consistency_criterion(self.cycle_report, self.real_report)
        total_cycle_loss = self.lambda_cycle * (cycle_loss_IRI + cycle_loss_RIR) + 1 * cycle_loss_IRI_MSE

        ############################################################################################

        total_gen_loss = total_adv_loss + total_cycle_loss

        losses = {
            "total_gen_loss": total_gen_loss,
            "total_adv_loss": total_adv_loss,
            "total_cycle_loss": total_cycle_loss,
            "adv_loss_IR": adv_loss_IR,
            "adv_loss_RI": adv_loss_RI,
            "cycle_loss_IRI_perceptul": cycle_loss_IRI_perceptul,
            "cycle_loss_IRI_MSE": cycle_loss_IRI_MSE,
            "cycle_loss_RIR": cycle_loss_RIR,
        }

        # add mode key to losses
        losses = {f"{mode}_{key}": value for key, value in losses.items()}
        self.log_dict(losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_gen_loss

    def image_discriminator_step(self, valid, fake, mode="train"):
        fake_img = self.buffer_images(self.fake_img)
        # calculate loss for discriminator
        ###########################################################################################
        # calculate on real data
        real_img_adv_loss = self.img_adv_criterion(self.image_discriminator(self.real_img), valid)
        # calculate on fake data
        fake_img_adv_loss = self.img_adv_criterion(self.image_discriminator(fake_img.detach()), fake)
        ###########################################################################################
        total_img_disc_loss = (real_img_adv_loss + fake_img_adv_loss) / 2

        losses = {
            "total_img_disc_loss": total_img_disc_loss,
            "real_img_adv_loss": real_img_adv_loss,
            "fake_img_adv_loss": fake_img_adv_loss,
        }

        # add mode key to losses
        losses = {f"{mode}_{key}": value for key, value in losses.items()}
        self.log_dict(losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return total_img_disc_loss

    def report_discriminator_step(self, valid, fake, mode="train"):
        fake_report = self.buffer_reports(self.fake_report)
        # calculate loss for discriminator
        ###########################################################################################
        # calculate on real data
        real_report_adv_loss = self.report_adv_criterion(self.report_discriminator(self.real_report), valid)
        # calculate on fake data
        fake_report_adv_loss = self.report_adv_criterion(self.report_discriminator(fake_report.detach()), fake)
        ###########################################################################################
        total_report_disc_loss = (real_report_adv_loss + fake_report_adv_loss) / 2

        losses = {
            "total_report_disc_loss": total_report_disc_loss,
            "real_report_adv_loss": real_report_adv_loss,
            "fake_report_adv_loss": fake_report_adv_loss,
        }

        # add mode key to losses
        losses = {f"{mode}_{key}": value for key, value in losses.items()}
        self.log_dict(losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_report_disc_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.real_img = batch['target'].float()
        self.real_report = batch['report'].float()
        batch_nmb = self.real_img.shape[0]

        z = Variable(torch.randn(batch_nmb, self.z_size)).float().to(self.device)

        # generate fake and valid samples
        valid_img = Tensor(np.ones((self.real_img.size(0), *self.image_discriminator.output_shape)))
        fake_img = Tensor(np.zeros((self.real_img.size(0), *self.image_discriminator.output_shape)))
        valid_report = Tensor(np.ones((self.real_report.size(0), *self.report_discriminator.output_shape)))
        fake_report = Tensor(np.zeros((self.real_report.size(0), *self.report_discriminator.output_shape)))

        # generate fake reports and images
        self.fake_report = self.report_generator(self.real_img)
        self.fake_img = self.image_generator(z, self.real_report)

        fake_reports = torch.sigmoid(self.fake_report)
        # fake_reports = torch.where(fake_reports > 0.5, 1, 0)

        # reconstruct images and reports
        self.cycle_img = self.image_generator(z, fake_reports)
        self.cycle_report = self.report_generator(self.real_img)

        cycle_reports = torch.sigmoid(self.cycle_report)
        cycle_reports = torch.where(cycle_reports > 0.5, 1.0, 0.0)

        # calculate metrics
        self.macro_accuracy['train'].update(cycle_reports, self.real_report)
        self.micro_accuracy['train'].update(cycle_reports, self.real_report)
        self.macro_f1['train'].update(cycle_reports, self.real_report)
        self.micro_f1['train'].update(cycle_reports, self.real_report)
        self.macro_precision['train'].update(cycle_reports, self.real_report)
        self.micro_precision['train'].update(cycle_reports, self.real_report)
        self.macro_recall['train'].update(cycle_reports, self.real_report)
        self.micro_recall['train'].update(cycle_reports, self.real_report)
        precision_overall = self.calculate_metrics_overall(cycle_reports, self.real_report, batch_nmb)
        
        metrics = {
            "macro_accuracy": self.macro_accuracy['train'],
            "micro_accuracy": self.micro_accuracy['train'],
            "macro_f1": self.macro_f1['train'],
            "micro_f1": self.micro_f1['train'],
            "macro_precision": self.macro_precision['train'],
            "micro_precision": self.micro_precision['train'],
            "macro_recall": self.macro_recall['train'],
            "micro_recall": self.micro_recall['train'],
            "precision_overall": precision_overall,
        }
        # add mode key to metrics
        metrics = {f"train_{key}": value for key, value in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        if batch_idx % 100 == 0 and (self.current_epoch % 10 == 0):
            if optimizer_idx == 0:
                self.log_images_on_cycle(batch_idx)
                self.log_reports_on_cycle(batch_idx)
                self.visualize_images(batch_idx)

        if optimizer_idx == 0 or optimizer_idx == 1:
            gen_tmp_loss = self.generator_step(valid_img, valid_report, mode="train")
            if gen_tmp_loss > self.gen_threshold:
                gen_loss = gen_tmp_loss
                return gen_loss

        elif optimizer_idx == 2:
            img_disc_tmp_loss = self.image_discriminator_step(valid_img, fake_img, mode="train")
            if img_disc_tmp_loss > self.disc_threshold:
                img_disc_loss = img_disc_tmp_loss
                return img_disc_loss

        elif optimizer_idx == 3:
            report_disc_tmp_loss = self.report_discriminator_step(valid_report, fake_report, mode="train")
            if report_disc_tmp_loss > self.disc_threshold:
                report_disc_loss = report_disc_tmp_loss
                return report_disc_loss

    # def training_epoch_end(self):
    #     # reset the metrics
    #     self.macro_accuracy['train'].reset()
    #     self.micro_accuracy['train'].reset()
    #     self.macro_f1['train'].reset()
    #     self.micro_f1['train'].reset()
    #     self.macro_precision['train'].reset()
    #     self.micro_precision['train'].reset()
    #     self.macro_recall['train'].reset()
    #     self.micro_recall['train'].reset()

    def validation_step(self, batch, batch_idx):
        self.real_img = batch['target'].float()
        self.real_report = batch['report'].float()
        batch_nmb = self.real_img.shape[0]

        true_reports = self.real_report
        self.fake_report = self.report_generator(self.real_img)
        self.fake_report = torch.sigmoid(self.fake_report)
        self.fake_report_0_1 = torch.where(self.fake_report > 0.5, 1.0, 0.0)

        # calculate metrics
        self.macro_accuracy['val'].update(self.fake_report_0_1, self.real_report)
        self.micro_accuracy['val'].update(self.fake_report_0_1, self.real_report)
        self.macro_f1['val'].update(self.fake_report_0_1, self.real_report)
        self.micro_f1['val'].update(self.fake_report_0_1, self.real_report)
        self.macro_precision['val'].update(self.fake_report_0_1, self.real_report)
        self.micro_precision['val'].update(self.fake_report_0_1, self.real_report)
        self.macro_recall['val'].update(self.fake_report_0_1, self.real_report)
        self.micro_recall['val'].update(self.fake_report_0_1, self.real_report)
        precision_overall = self.calculate_metrics_overall(self.fake_report_0_1, self.real_report, batch_nmb)

        metrics = {
            "macro_accuracy": self.macro_accuracy['val'],
            "micro_accuracy": self.micro_accuracy['val'],
            "macro_f1": self.macro_f1['val'],
            "micro_f1": self.micro_f1['val'],
            "macro_precision": self.macro_precision['val'],
            "micro_precision": self.micro_precision['val'],
            "macro_recall": self.macro_recall['val'],
            "micro_recall": self.micro_recall['val'],
            "precision_overall": precision_overall,
        }

        # add mode key to metrics
        metrics = {f"val_{key}": value for key, value in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        # Shuffle data here as paired data is needed for above part
        indices = torch.randperm(batch_nmb)
        self.real_report = self.real_report[indices]
        # Calculating losses for CycleGAN
        z = Variable(torch.randn(batch_nmb, self.z_size)).float().to(self.device)

        # generate fake and valid samples
        valid_img = Tensor(np.ones((self.real_img.size(0), *self.image_discriminator.output_shape)))
        fake_img = Tensor(np.zeros((self.real_img.size(0), *self.image_discriminator.output_shape)))
        valid_report = Tensor(np.ones((self.real_report.size(0), *self.report_discriminator.output_shape)))
        fake_report = Tensor(np.zeros((self.real_report.size(0), *self.report_discriminator.output_shape)))

        # generate fake reports and images
        self.fake_img = self.image_generator(z, self.real_report)
        # reconstruct reports and images
        self.cycle_report = self.report_generator(self.fake_img)
        self.cycle_img = self.image_generator(z, self.fake_report_0_1)

        cycle_report = torch.sigmoid(self.cycle_report)
        cycle_report = torch.where(cycle_report > 0.5, 1.0, 0.0)
        
        # calculate metrics
        self.macro_accuracy['val'].update(cycle_report, self.real_report)
        self.micro_accuracy['val'].update(cycle_report, self.real_report)
        self.macro_f1['val'].update(cycle_report, self.real_report)
        self.micro_f1['val'].update(cycle_report, self.real_report)
        self.macro_precision['val'].update(cycle_report, self.real_report)
        self.micro_precision['val'].update(cycle_report, self.real_report)
        self.macro_recall['val'].update(cycle_report, self.real_report)
        self.micro_recall['val'].update(cycle_report, self.real_report)
        precision_overall = self.calculate_metrics_overall(cycle_report, self.real_report, batch_nmb)

        metrics = {
            "macro_accuracy": self.macro_accuracy['val'],
            "micro_accuracy": self.micro_accuracy['val'],
            "macro_f1": self.macro_f1['val'],
            "micro_f1": self.micro_f1['val'],
            "macro_precision": self.macro_precision['val'],
            "micro_precision": self.micro_precision['val'],
            "macro_recall": self.macro_recall['val'],
            "micro_recall": self.micro_recall['val'],
            "precision_overall": precision_overall,
        }
        # add mode key to metrics
        metrics = {f"val_{key}": value for key, value in metrics.items()}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        val_gen_loss = self.generator_step(valid_img, valid_report, mode="val")
        val_img_disc_loss = self.image_discriminator_step(valid_img, fake_img, mode="val")
        val_report_disc_loss = self.report_discriminator_step(valid_report, fake_report, mode="val")

    # def validation_epoch_end(self):
    #     # reset the metrics
    #     self.macro_accuracy['val'].reset()
    #     self.micro_accuracy['val'].reset()
    #     self.macro_f1['val'].reset()
    #     self.micro_f1['val'].reset()
    #     self.macro_precision['val'].reset()
    #     self.micro_precision['val'].reset()
    #     self.macro_recall['val'].reset()
    #     self.micro_recall['val'].reset()


    def calculate_metrics_overall(self, preds, targets, batch_nmb):
        exact_matches = torch.all(preds == targets, dim=1)
        true_positives = torch.sum(exact_matches).item()

        precision = true_positives / batch_nmb

        return precision

    def log_images_on_cycle(self, batch_idx):

        cycle_img_1 = self.cycle_img[0]
        real_img_1 = self.real_img[0]
        fake_img_1 = self.fake_img[0]

        cycle_img_tensor = cycle_img_1
        real_img_tensor = real_img_1
        fake_img_tensor = fake_img_1

        step = self.current_epoch * batch_idx + batch_idx

        self.logger.experiment.add_image(f"On step cycle img", cycle_img_tensor, step, dataformats='CHW')
        self.logger.experiment.add_image(f"On step real img", real_img_tensor, step, dataformats='CHW')
        self.logger.experiment.add_image(f"On step fake_img", fake_img_tensor, step, dataformats='CHW')
    
    def log_reports_on_cycle(self, batch_idx):
        real_report = self.real_report[0]
        cycle_report = self.cycle_report[0]
        # Process the generated report
        real_report = real_report.cpu().detach()
        real_report = torch.sigmoid(real_report)
        real_report = (real_report > 0.5).int()
        real_report_tensor = real_report
        report_text_labels = [self.opt['dataset']['chexpert_labels'][idx] for idx, val in enumerate(real_report) if
                              val == 1]
        report_text_real = ', '.join(report_text_labels)
        report_text_real_tensor = ', '.join([str(val) for val in real_report_tensor])

        generated_report = cycle_report.cpu().detach()
        generated_report_logits = generated_report
        generated_report = torch.sigmoid(generated_report)
        generated_report = (generated_report > 0.5).int()
        generated_report_tensor = generated_report
        report_text_labels_cycle = [self.opt['dataset']['chexpert_labels'][idx] for idx, val in enumerate(generated_report) if
                              val == 1]
        report_text_cycle = ', '.join(report_text_labels_cycle)
        report_text_cycle_logits = ', '.join([str(val) for val in generated_report_logits])
        report_text_cycle_tensor = ', '.join([str(val) for val in generated_report_tensor])

        step = self.current_epoch * batch_idx + batch_idx

        self.logger.experiment.add_text(f"On step cycle report", report_text_real, step)
        self.logger.experiment.add_text(f"On step real report", report_text_cycle, step)

        self.logger.experiment.add_text(f"On step cycle report logits", report_text_cycle_logits, step)
        self.logger.experiment.add_text(f"On step cycle report tensor", report_text_cycle_tensor, step)
        self.logger.experiment.add_text(f"On step real report tensor", report_text_real_tensor, step)

   def visualize_images(self, batch_idx):
        real_img = self.convert_tensor_to_image(self.real_img[0])
        plt.imshow(real_img)
        plt.axis('off')
        plt.show()
        cycle_img = self.convert_tensor_to_image(self.cycle_img[0])
        plt.imshow(cycle_img)
        plt.axis('off')
        plt.show()

    def save_images(self, batch_idx):
        # Create the folder if it does not exist
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        # Process and save the real image
        real_image = self.convert_tensor_to_image(self.real_img[0])
        real_image_path = os.path.join(self.save_folder, f'real_image_{batch_idx}.png')
        real_image.save(real_image_path)

        # Process and save the cycle image
        cycle_image = self.convert_tensor_to_image(self.cycle_img[0])
        cycle_image_path = os.path.join(self.save_folder, f'cycle_image_{batch_idx}.png')
        cycle_image.save(cycle_image_path)

    def convert_tensor_to_image(self, tensor):
        # Denormalize and convert to PIL Image
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        denorm = tensor.clone().cpu().detach()
        for t, m, s in zip(denorm, mean, std):
            t.mul_(s).add_(m)
        denorm = denorm.numpy().transpose(1, 2, 0)
        denorm = np.clip(denorm, 0, 1)
        return Image.fromarray((denorm * 255).astype('uint8'))