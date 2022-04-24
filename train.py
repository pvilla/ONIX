"""
    Model training
"""

import os
import time
import torch
from models.options import ParamOptions
from models.trainer import TrainModel

if __name__ == "__main__":
    opt = ParamOptions().parse()
    model = TrainModel(opt)
    model.init_model()
    destination = f"{opt.run_path}/{opt.run_name}"
    print(destination)
    if opt.load_pretrain:
        model.load_trained_models(opt.model_path, opt.load_epoch)  # Pretrain
    model.save_parameters()
    for epoch in range(opt.num_epochs):
        now = time.time()
        model.update_parameters(epoch)
        for i, train_data in enumerate(model.train_loader):
            model.set_input(train_data)
            model.optimization()  # The original optimization. Could be biased with gradient sampling
            if i % opt.print_loss_freq_iter == opt.print_loss_freq_iter - 1:
                losses = model.get_current_losses()
                model.print_current_losses(epoch, i, losses)
        if (
            epoch == 0
            or epoch % opt.save_plot_freq_epoch == opt.save_plot_freq_epoch - 1
        ):
            with torch.no_grad():
                model.generate_3D = True
                for k, test_data in enumerate(model.test_loader):
                    if k == 0:
                        model.set_input(test_data[:2])
                        model.validation()
                        losses = model.get_val_losses()
                        model.print_val_losses(epoch, i, losses)
                        model.visual_iter(epoch, i, test_data[2].item())
                model.generate_3D = False
        else:
            pass
        if epoch % opt.save_model_freq_epoch == opt.save_model_freq_epoch - 1:
            model.save_models(epoch)
        now = time.time() - now
        print(
            f"training time for {opt.run_name} epoch {epoch+1}: {now//60} min {round(now - now//60*60)} s"
        )
    print(destination)
