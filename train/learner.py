import torch
import numpy as np
import json
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Optional
import os


def nested_map(struct, map_fn):
    """This is for trasfering into cuda device"""
    if isinstance(struct, tuple):
        return tuple(nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


class DiffproLearner:
    def __init__(
        self, output_dir, model, train_dl, val_dl, optimizer, params
    ):
        # model output
        self.output_dir = output_dir
        self.log_dir = f"{output_dir}/logs"
        self.checkpoint_dir = f"{output_dir}/chkpts"
        # model (architecture and loss)
        self.model = model                  
        # data loader
        self.train_dl = train_dl
        self.val_dl = val_dl
        # optimizer
        self.optimizer = optimizer
        # what is this ????
        self.params = params
        # current time recoder 
        self.step = 0
        self.epoch = 0
        self.grad_norm = 0.
        # other information
        self.summary_writer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast = torch.cuda.amp.autocast(enabled=params.fp16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=params.fp16)

        self.best_val_loss = torch.tensor([1e10], device=self.device)

        # restore if directory exists
        if os.path.exists(self.output_dir):
            self.restore_from_checkpoint()
        else:
            os.makedirs(self.output_dir)
            os.makedirs(self.log_dir)
            os.makedirs(self.checkpoint_dir)
            with open(f"{output_dir}/params.json", "w") as params_file:
                json.dump(self.params, params_file)

        print(json.dumps(self.params, sort_keys=True, indent=4))

    def _write_summary(self, losses: dict, scheduled_params: Optional[dict], type):
        """type: train or val"""
        summary_losses = losses
        summary_losses["grad_norm"] = self.grad_norm
        if scheduled_params is not None:
            for k, v in scheduled_params.items():
                summary_losses[f"sched_{k}"] = v
        writer = self.summary_writer or SummaryWriter(
            self.log_dir, purge_step=self.step
        )
        writer.add_scalars(type, summary_losses, self.step)
        writer.flush()
        self.summary_writer = writer

    def state_dict(self):
        # state dictionary
        model_state = self.model.state_dict()
        return {
            "step": self.step,
            "epoch": self.epoch,
            "model":
                {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in model_state.items()
                },
            "optimizer":
                {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in self.optimizer.state_dict().items()
                },
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scaler.load_state_dict(state_dict["scaler"])

    def restore_from_checkpoint(self, fname="weights"):
        try:
            fpath = f"{self.checkpoint_dir}/{fname}.pt"
            checkpoint = torch.load(fpath)
            self.load_state_dict(checkpoint)
            print(f"Restored from checkpoint {fpath} --> {fname}-{self.epoch}.pt!")
            return True
        except FileNotFoundError:
            print("No checkpoint found. Starting from scratch...")
            return False

    def _link_checkpoint(self, save_name, link_fpath):
        if os.path.islink(link_fpath):
            os.unlink(link_fpath)
        os.symlink(save_name, link_fpath)

    def save_to_checkpoint(self, fname="weights", is_best=False):
        save_name = f"{fname}-{self.epoch}.pt"
        save_fpath = f"{self.checkpoint_dir}/{save_name}"
        link_best_fpath = f"{self.checkpoint_dir}/{fname}_best.pt"
        link_fpath = f"{self.checkpoint_dir}/{fname}.pt"
        torch.save(self.state_dict(), save_fpath)
        self._link_checkpoint(save_name, link_fpath)
        if is_best:
            self._link_checkpoint(save_name, link_best_fpath)

    def train(self, max_epoch=None, null_rhythm_prob=0.0):
        self.model.train()

        while True:
            self.epoch = self.step // len(self.train_dl)
            if max_epoch is not None and self.epoch >= max_epoch:
                return

            for batch in tqdm(self.train_dl, desc=f"Epoch {self.epoch}"):
                '''batch shape: (batch_size * channels * length * height)
                possible channels:
                channels = 6: acc_onset, acc_sustain, chd_onset_rhythm, chd_sustain_rhythm, melody_onset, melody_sustain
                channels = 8: acc_onset, acc_sustain, chd_onset_rhythm, chd_sustain_rhythm, chd_onset_null, chd_sustain_null, melody_onset, melody_sustain
                '''
                batch = nested_map(
                    batch, lambda x: x.to(self.device)
                    if isinstance(x, torch.Tensor) else x
                )
                # Ensure batch is a tensor
                if not isinstance(batch, torch.Tensor):
                    raise TypeError(f"Expected batch to be a tensor, got {type(batch)}")
                batch: torch.Tensor = batch  # type annotation
                
            
                assert batch.shape[1] in [6,8]
                if batch.shape[1]==6:
                    # randomly change the rhythm condition into null rhythm
                    random_tensor = torch.rand(batch.shape[0]).reshape(-1,1,1,1).repeat((1,4,batch.shape[2],batch.shape[3])).to(self.device)
                    use_null_rhythm = torch.where(random_tensor < null_rhythm_prob, torch.tensor(1.0), torch.tensor(0.0))
                    edited_batch = torch.where(use_null_rhythm>0.5, batch[:,[0,1,4,5],:,:], batch[:,[0,1,2,3],:,:])
                    batch = edited_batch # batch_size * 4 * length * height
                else: # batch.shape[1]==8
                    # randomly change the rhythm condition into null rhythm
                    random_tensor_rhythm = torch.rand(batch.shape[0]).reshape(-1,1,1,1).repeat((1,2,batch.shape[2],batch.shape[3])).to(self.device)
                    use_null_rhythm = torch.where(random_tensor_rhythm < null_rhythm_prob, torch.tensor(1.0), torch.tensor(0.0))

                    random_tensor_melody = torch.rand(batch.shape[0]).reshape(-1,1,1,1).repeat((1,2,batch.shape[2],batch.shape[3])).to(self.device)
                    use_null_melody = torch.where(random_tensor_melody < null_rhythm_prob, torch.tensor(1.0), torch.tensor(0.0))
                    null_leadsheet = -torch.ones_like(batch[:,-2:,:,:])

                    chd_roll_to_use = torch.where(use_null_rhythm>0.5, batch[:,[4,5],:,:], batch[:,[2,3],:,:])
                    melody_roll_to_use = torch.where(use_null_melody>0.5, null_leadsheet, batch[:,[6,7],:,:])
                    
                    batch = torch.concat([batch,null_leadsheet], axis=1)
                    batch = torch.cat([batch[:,:2,:,:], chd_roll_to_use, melody_roll_to_use], axis=1) # batch_size * 6 * length * height
                    assert batch.shape[1]==6

                losses, scheduled_params = self.train_step(batch)
                # check NaN
                for loss_value in list(losses.values()):
                    if isinstance(loss_value,
                                  torch.Tensor) and torch.isnan(loss_value).any():
                        raise RuntimeError(
                            f"Detected NaN loss at step {self.step}, epoch {self.epoch}"
                        )
                if self.step % 50 == 0:
                    self._write_summary(losses, scheduled_params, "train")
                if self.step % 5000 == 0 and self.step != 0 \
                    and self.epoch != 0:
                    self.valid(null_rhythm_prob=null_rhythm_prob)
                self.step += 1

            # valid
            self.valid(null_rhythm_prob=null_rhythm_prob)

    def valid(self, null_rhythm_prob=0.0):
        # self.model.eval()
        losses = None
        for batch in self.val_dl:
            batch = nested_map(
                batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x
            )
            # Ensure batch is a tensor
            if not isinstance(batch, torch.Tensor):
                raise TypeError(f"Expected batch to be a tensor, got {type(batch)}")
            batch: torch.Tensor = batch  # type annotation

            if batch.shape[1]==6:
                # randomly change the rhythm condition into null rhythm
                random_tensor = torch.rand(batch.shape[0]).reshape(-1,1,1,1).repeat((1,4,batch.shape[2],batch.shape[3])).to(self.device)
                use_null_rhythm = torch.where(random_tensor < null_rhythm_prob, torch.tensor(1.0), torch.tensor(0.0))
                edited_batch = torch.where(use_null_rhythm>0.5, batch[:,[0,1,4,5],:,:], batch[:,[0,1,2,3],:,:])
                batch = edited_batch # batch_size * 4 * length * height
            else: # batch.shape[1]==8
                # randomly change the rhythm condition into null rhythm
                random_tensor_rhythm = torch.rand(batch.shape[0]).reshape(-1,1,1,1).repeat((1,2,batch.shape[2],batch.shape[3])).to(self.device)
                use_null_rhythm = torch.where(random_tensor_rhythm < null_rhythm_prob, torch.tensor(1.0), torch.tensor(0.0))

                random_tensor_melody = torch.rand(batch.shape[0]).reshape(-1,1,1,1).repeat((1,2,batch.shape[2],batch.shape[3])).to(self.device)
                use_null_melody = torch.where(random_tensor_melody < null_rhythm_prob, torch.tensor(1.0), torch.tensor(0.0))
                null_leadsheet = -torch.ones_like(batch[:,-2:,:,:])

                chd_roll_to_use = torch.where(use_null_rhythm>0.5, batch[:,[4,5],:,:], batch[:,[2,3],:,:])
                melody_roll_to_use = torch.where(use_null_melody>0.5, null_leadsheet, batch[:,[6,7],:,:])
                
                batch = torch.concat([batch,null_leadsheet], axis=1)
                batch = torch.cat([batch[:,:2,:,:], chd_roll_to_use, melody_roll_to_use], axis=1) # batch_size * 6 * length * height
                assert batch.shape[1]==6

            # print(batch.shape)
            current_losses, _ = self.val_step(batch)
            losses = losses or current_losses
            for k, v in current_losses.items():
                losses[k] += v
        assert losses is not None
        for k, v in losses.items():
            losses[k] /= len(self.val_dl)
        print(losses)
        self._write_summary(losses, None, "val")

        if self.best_val_loss >= losses["loss"]:
            self.best_val_loss = losses["loss"]
            self.save_to_checkpoint(is_best=True)
        else:
            self.save_to_checkpoint(is_best=False)

    def train_step(self, batch):
        # people say this is the better way to set zero grad
        # instead of self.optimizer.zero_grad()
        for param in self.model.parameters():
            param.grad = None

        # here forward the model
        with self.autocast:
            scheduled_params = None
            loss_dict = self.model.get_loss_dict(batch, self.step)

        loss = loss_dict["loss"]
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss_dict, scheduled_params

    def val_step(self, batch):
        with torch.no_grad():
            with self.autocast:
                scheduled_params = None
                loss_dict = self.model.get_loss_dict(batch, self.step)

        return loss_dict, scheduled_params
