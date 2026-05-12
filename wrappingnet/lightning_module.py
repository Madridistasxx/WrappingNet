import torch
from hydra.utils import get_method, instantiate
from pytorch_lightning.core import LightningModule

import wrappingnet.losses as losses
import wrappingnet.utils as utils


class WrappingNetLightning(LightningModule):
    def __init__(
        self,
        model,
        optimizer_cfg,
        scheduler_cfg=None,
        batch_size=1,
        epochs_sphere=200,
        lmbda=1.0,
        norm=False,
        distortion_loss="wrappingnet.losses.multiscale_l2",
        eval_loss="wrappingnet.losses.l2",
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.batch_size = batch_size
        self.epochs_sphere = epochs_sphere
        self.lmbda = lmbda
        self.norm = norm

        self.distortion_func = get_method(distortion_loss)
        self.distortion_eval = get_method(eval_loss)

    def training_step(self, data, batch_idx):
        if self.norm:
            data.pos = utils.normalize_pos(data.pos, 1)

        pos_base, faces_base = utils.get_base_mesh(data.pos, data.face.T)

        if self.current_epoch < self.epochs_sphere:
            pos_sphere = self.model.make_sphere(pos_base, faces_base)
            loss = losses.base_loss(pos_sphere, scale=10)

            self.log_dict(
                {
                    "train_base": loss.item(),
                    "step": self.current_epoch * 1.0,
                },
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )

            return loss

        pos_list, face_list, _ = self.model(data.pos, data.face.T, pos_base)

        rate = torch.tensor(0.0, device=data.pos.device)

        distortion_loss = self.distortion_func(
            pos_list,
            face_list,
            data.pos,
            data.face.T,
        )

        chamfer_loss = losses.chamfer(pos_list[-1], data.pos)

        loss = distortion_loss

        self.log_dict(
            {
                "train_rate": rate.item(),
                "train_distortion": distortion_loss.item(),
                "train_chamfer": chamfer_loss.item(),
                "train_loss": loss.item(),
                "step": self.current_epoch * 1.0,
            },
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

        return loss

    def validation_step(self, data, batch_idx):
        if self.norm:
            data.pos = utils.normalize_pos(data.pos, 1)

        pos_base, faces_base = utils.get_base_mesh(data.pos, data.face.T)

        if self.current_epoch < self.epochs_sphere:
            pos_sphere = self.model.make_sphere(pos_base, faces_base)
            loss = losses.base_loss(pos_sphere, scale=10)

            self.log_dict(
                {
                    "val_base": loss.item(),
                    "step": self.current_epoch * 1.0,
                },
                sync_dist=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )

            return loss

        pos_list, face_list, _ = self.model(data.pos, data.face.T, pos_base)

        rate = torch.tensor(0.0, device=data.pos.device)

        distortion_loss = self.distortion_eval(
            pos_list,
            face_list,
            data.pos,
            data.face.T,
        )

        chamfer_loss = losses.chamfer(pos_list[-1], data.pos)

        loss = rate + self.lmbda * distortion_loss

        self.log_dict(
            {
                "val_rate": rate.item(),
                "val_distortion": distortion_loss.item(),
                "val_chamfer": chamfer_loss.item(),
                "val_loss": loss.item(),
                "step": self.current_epoch * 1.0,
            },
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

        return loss

    def configure_optimizers(self):
        optimizer = instantiate(
            self.optimizer_cfg,
            params=self.model.parameters(),
        )

        if self.scheduler_cfg is None:
            return optimizer

        scheduler = instantiate(
            self.scheduler_cfg,
            optimizer=optimizer,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }