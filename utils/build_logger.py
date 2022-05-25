from pytorch_lightning.loggers import CSVLogger


def build_logger(logger, model, configs):
    # if logger == "Neptune":
    #     return neptune_logger
    if logger == 'csv':
        return CSVLogger("logs", name="csvlogger")
    else:
        return True
