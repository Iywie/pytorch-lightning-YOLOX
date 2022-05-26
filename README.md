# pl_YOLOX
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
