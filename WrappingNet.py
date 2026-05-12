import os
from types import SimpleNamespace

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer

import wrappingnet.dataloaders as dataloaders
import wrappingnet.losses as losses
import wrappingnet.models as models
import wrappingnet.utils as utils


def build_legacy_args(cfg: DictConfig, model=None):
    """
    兼容旧代码：
    - models.get_model(args) 需要 args.model_name / args.latent_dim
    - dataloaders.get_data_lightning(args) 需要 args.data_name / args.data_root / args.batch_size
    - WrappingNetLightning(**vars(args)) 需要 flat namespace
    """
    return SimpleNamespace(
        gpus=list(cfg.hardware.gpus),
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        epochs=cfg.training.epochs,
        epochs_sphere=cfg.training.epochs_sphere,
        latent_dim=cfg.model.latent_dim,
        data_name=cfg.data.name,
        model_name=cfg.model.name,
        data_root=cfg.paths.data_root,
        loss_func=cfg.training.loss_func,
        pretrain=cfg.training.pretrain,
        load_make_sphere=cfg.training.load_make_sphere,
        norm10=cfg.training.norm10,
        norm=cfg.training.norm,
        lmbda=cfg.training.lmbda,
        model=model,
    )


def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    args = build_legacy_args(cfg)

    extra = args.model_name
    model = models.get_model(args)

    if args.pretrain:
        ckpt_path = os.path.join(
            cfg.paths.train_dir,
            f"MeshAE_{args.loss_func}_{args.data_name}_d{args.latent_dim}{extra}.ckpt",
        )
        saved = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(saved)

    elif args.load_make_sphere:
        print("LOAD MAKESPHERE")
        ckpt_path = os.path.join(
            cfg.paths.make_sphere_dir,
            f"make_sphere_{args.data_name}.ckpt",
        )
        saved = torch.load(ckpt_path, map_location="cpu")
        model.make_sphere.load_state_dict(saved)

    args = build_legacy_args(cfg, model=model)
    lightning_model = WrappingNetLightning(**vars(args))

    trainer = Trainer(
        accelerator=cfg.hardware.accelerator,
        devices=len(cfg.hardware.gpus),
        max_epochs=cfg.training.epochs,
        callbacks=[LearningRateMonitor(logging_interval="epoch")],
        strategy=cfg.trainer.strategy,
    )

    datamodule = dataloaders.get_data_lightning(args)
    trainer.fit(lightning_model, datamodule)

    os.makedirs(cfg.paths.train_dir, exist_ok=True)
    save_path = os.path.join(
        cfg.paths.train_dir,
        f"MeshAE_{args.loss_func}_{args.data_name}_d{args.latent_dim}{extra}.ckpt",
    )
    torch.save(lightning_model.model.state_dict(), save_path)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()