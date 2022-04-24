#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 13:44:04 2021

Generate 3D models
This is used to generate a model with the size of the original input, e.g., for H x W inputs, this will generate a H x W x Depth volume.

@author: Yuhe Zhang
"""
import time
import torch
from models.eval_options import ParamOptions
from models.trainer import TrainModel

if __name__ == "__main__":
    opt = ParamOptions().parse()
    model = TrainModel(opt)
    model.is_train = False
    model.generate_3D = True
    model.init_model()
    model.load_trained_models(opt.model_path, opt.load_epoch)

    now = time.time()
    with torch.no_grad():
        for k, test_data in enumerate(model.test_loader):
            if k == 0:
                model.set_input(test_data[:2])
                model.validation()
                model.visual_iter(0, k, test_data[2].item())
        now = time.time() - now
        print(f"validation time: {now//60} min {round(now - now//60*60)} s")
