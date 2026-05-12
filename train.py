import os

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

torch.set_num_threads(4)
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed, workers=True)

    model = instantiate(cfg.model.net)

    lightning_model = instantiate(
        {
            "_target_": "wrappingnet.lightning_module.WrappingNetLightning",
            "model": model,
            "optimizer_cfg": cfg.optimizer,
            "scheduler_cfg": cfg.scheduler,
            "batch_size": cfg.training.batch_size,
            "epochs_sphere": cfg.training.epochs_sphere,
            "lmbda": cfg.training.lmbda,
            "norm": cfg.training.norm,
            "distortion_loss": cfg.loss.distortion,
            "eval_loss": cfg.loss.eval,
        },
        _recursive_=False,
    )

    if cfg.checkpoint.pretrain_path is not None:
        ckpt_path = to_absolute_path(cfg.checkpoint.pretrain_path)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        lightning_model.model.load_state_dict(state_dict)

    if cfg.checkpoint.make_sphere_path is not None:
        ckpt_path = to_absolute_path(cfg.checkpoint.make_sphere_path)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        lightning_model.model.make_sphere.load_state_dict(state_dict)

    datamodule = instantiate(cfg.data.datamodule)

    trainer = instantiate(cfg.trainer)

    trainer.fit(lightning_model, datamodule)

    if cfg.checkpoint.save:
        save_dir = to_absolute_path(cfg.paths.trained_dir)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, cfg.checkpoint.save_name)
        torch.save(lightning_model.model.state_dict(), save_path)

        print(f"Saved checkpoint to: {save_path}")


if __name__ == "__main__":
    main()