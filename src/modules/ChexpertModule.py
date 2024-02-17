import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from utils._prepare_data import DataHandler
from data.Chexpert import Chexpert
import random


class ChexpertDataModule(pl.LightningDataModule):

    def __init__(self, opt, processor:DataHandler):
        super().__init__()
        self.opt = opt
        self.processor = processor
        self.batch_size = self.opt["batch_size"]
        self.num_workers = self.opt["num_workers"]
        self.train_records = []
        self.val_records = []
        self.test_records = []

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        # Called on every process in DDP
        self.records = self.processor.records
        # create train val and test records based on the split key value of each record
        if stage in {"fit", None}:
            train_records = []
            val_records = []
            for record in self.records:
                if record["split"] == "train":
                    train_records.append(record)
                elif record["split"] == "val":
                    val_records.append(record)
                else:
                    raise Exception("Invalid split in training : " + record["split"])
                
            self.train_records = Chexpert(self.opt, self.processor, train_records, training=True)
            self.val_records = Chexpert(self.opt, self.processor, val_records, training=False)

        elif stage in {"test", None}:
            self.test_records = Chexpert(self.opt, self.processor, self.records, training=False)

        else:
            raise ValueError(f"Stage {stage} not recognized")
        

    def train_dataloader(self):
        return DataLoader(self.train_records,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, 
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_records,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, 
                          shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_records,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, 
                          shuffle=False)
    


