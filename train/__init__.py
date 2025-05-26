import torch
import json
import os
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from .learner import DiffproLearner


class TrainConfig:

    model: torch.nn.Module
    train_dl: DataLoader
    val_dl: DataLoader
    optimizer: Optimizer

    def __init__(self, params, output_dir) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = params
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

