import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from model import init_ldm_model, init_diff_pro_sdf, Diffpro_SDF
from data.dataset_loading import load_datasets, create_dataloader
from datetime import datetime
from train.learner import DiffproLearner
from . import *

# Determine the absolute path to the external folder
current_directory = os.path.dirname(os.path.abspath(__file__))
external_directory = os.path.abspath(os.path.join(current_directory, '../data'))

# Add the external folder to sys.path
sys.path.append(external_directory)

class TrainConfig:

    model: torch.nn.Module
    train_dl: DataLoader
    val_dl: DataLoader
    optimizer: Optimizer

    def __init__(self, params, param_scheduler, output_dir) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params
        self.param_scheduler = param_scheduler
        self.output_dir = output_dir

    def train(self, null_rhythm_prob=0.0):
        # collect and display total parameters
        total_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_parameters}")

        # dealing with the output storing
        output_dir = self.output_dir
        if os.path.exists(f"{output_dir}/chkpts/weights.pt"):
            print("Checkpoint already exists.")
            if input("Resume training? (y/n)") != "y":
                return
        else:
            output_dir = f"{output_dir}/{datetime.now().strftime('%m-%d_%H%M%S')}"
            print(f"Creating new log folder as {output_dir}")

        # prepare the learner structure and parameters
        learner = DiffproLearner(
            output_dir, self.model, self.train_dl, self.val_dl, self.optimizer,
            self.params
        )
        learner.train(max_epoch=self.params.max_epoch, null_rhythm_prob=null_rhythm_prob)
        
class LdmTrainConfig(TrainConfig):

    def __init__(self, params, output_dir, debug_mode=False, data_format="separate_melody_accompaniment", load_chkpt_from=None) -> None:
        super().__init__(params, None, output_dir)
        self.debug_mode = debug_mode
        #self.use_autoreg_cond = use_autoreg_cond
        #self.use_external_cond = use_external_cond
        #self.mask_background = mask_background
        #self.random_pitch_aug = random_pitch_aug

        # create model
        self.ldm_model = init_ldm_model(params, debug_mode)
        if load_chkpt_from is not None:
            self.model = Diffpro_SDF.load_trained(self.ldm_model, load_chkpt_from).to(self.device)
        else:
            self.model = init_diff_pro_sdf(self.ldm_model, params, self.device)


        # Create dataloader
        train_set = load_datasets(data_format=data_format)
        self.train_dl = create_dataloader(params.batch_size, train_set)
        self.val_dl = create_dataloader(params.batch_size, train_set[0:100]) # NOTE: we temporarily use the first 100 samples in train_set for validation, you can define your own valid dataloader

        # Create optimizer4
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.learning_rate
        )
