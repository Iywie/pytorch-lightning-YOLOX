# pytorch-lightning-YOLOX
YOLOX implemented by pytorch lightning, a simpler expression of pytorch

### train
`python train.py -c configs/yolox_s.yaml`

### COCO dataset
```
The _data_dir_ of configs  
├── annotations  
│   ├── train.json  
│   ├── val.json  
├── _train  
│   ├── <training images>  
├── _val  
│   ├── <validation images>  
```

### Pytorch Lightning Trainer of train.py 
The parameters are important

### The structure of our YOLOX
    class LitYOLOX(LightningModule):
        def __init__(self, cfgs)
        
        def on_train_start(self)
        
        def training_step(self, batch, batch_idx)
        
        def validation_step(self, batch, batch_idx)
        
        def validation_epoch_end(self, val_step_outputs)
        
        def configure_optimizers(self)
        
        def on_train_end(self)

