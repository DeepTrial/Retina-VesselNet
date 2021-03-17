import json

cfg_file=open("cfg.json","rb")
cfgs=json.load(cfg_file)


dataset_cfg=cfgs["dataset"]
sample_cfg=cfgs["sample"]
train_cfg=cfgs["training"]
test_cfg=cfgs["test"]