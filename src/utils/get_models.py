from models.ARK import ARKModel
from models.BioViL import BioViL
from losses.Test_loss import ClassificationLoss
from models.Discriminator import ImageDiscriminator, ReportDiscriminator

class Networks:
    def __init__(self, opt):
        self.opt = opt
        self.num_classes = opt['dataset']['num_classes']
        self.input_size = opt["dataset"]["input_size"]
        self.z_size = opt['image_generator']['z_size']

    def get_ark_model(self):
        """
        Get ARKModel
        """
        ark_model = ARKModel(num_classes=self.num_classes,
                            ark_pretrained_path=env_settings.PRETRAINED_PATH_ARK)

        return ark_model

    def get_biovil(self):
        return BioViL(embedding_size=self.opt["report_generator"]["embedding_size"],
                          num_classes=self.num_classes,
                          hidden_1=self.opt["report_generator"]["classification_head_hidden1"],
                          hidden_2=self.opt["report_generator"]["classification_head_hidden2"],
                          dropout_rate=self.opt["report_generator"]["dropout_prob"]
                    )

    def get_cosine_similarity(self):
        return ClassificationLoss(env_settings.MASTER_LIST[self.['dataset']['data_imputation']])

    
    def get_cgan(self):
        return cGANconv(z_size=self.z_size, img_size=self.input_size, class_num=self.num_classes,
                           img_channels=self.opt["image_discriminator"]["channels"])

    def get_image_discriminator(self):
        return ImageDiscriminator(input_shape=(self.opt['image_discriminator']['channels'], 
                                               self.opt['image_discriminator']['img_height'],
                                               self.opt['image_discriminator']['img_width'])
                                 )
    def get_report_discriminator(self):
        return ReportDiscriminator(input_dim=self.num_classes)


def load_networks(opt):

    networks = Networks(opt)

    if opt['report_generator']['image_encoder_model'].lower() == 'biovil':
        report_generator = networks.get_biovil()
    elif opt['report_generator']['image_encoder_model'].lower() == 'ark':
        report_generator = networks.get_ark_model()
    else:
        raise ValueError("Report generator model not found")

    if opt["report_discriminator"]["model"].lower() == "cosine_similarity":
        report_discriminator = networks.get_cosine_similarity()
    else:
        report_discriminator = networks.get_report_discriminator()

    if opt['image_generator']['report_encoder_model'].lower() == 'cgan':
        image_generator = networks.get_cgan()
    else:
        raise ValueError("Report generator model not found")

    image_discriminator = networks.get_image_discriminator()

    return report_generator, report_discriminator, image_generator, image_discriminator
