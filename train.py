from plyolox import LitYOLOX
from pldataset_coco import COCODataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import DeviceStatsMonitor
from utils.defaults import argument_parser, load_config
from utils.build_logger import build_logger


def main():
    args = argument_parser().parse_args()
    configs = load_config(args.cfg)

    model = LitYOLOX(configs)
    data = COCODataModule(configs)

    logger = build_logger(args.logger, model, configs)

    device_stats = DeviceStatsMonitor()
    seed_everything(96, workers=True)

    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer
    trainer = Trainer(
        gpus=1,
        max_epochs=300,
        check_val_every_n_epoch=5,
        log_every_n_steps=10,
        enable_progress_bar=True,
        logger=logger,
        # precision=16,
        # amp_backend="apex",
        # amp_level=01,
        # auto_lr_find=True,
        # benchmark=False,
        # callbacks=[device_stats],
        # default_root_dir="lightning_logs",
        # detect_anomaly=True,
        # limit_train_batches=3,
        # limit_val_batches=2,
        # reload_dataloaders_every_n_epochs=10,
    )

    trainer.fit(model, datamodule=data)
    # trainer.tune(model, datamodule=data)
    # trainer.validate(model, datamodule=data)
    # trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
