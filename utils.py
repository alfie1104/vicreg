import os
import torch

def save_model(model, optimizer, epoch, name, encoder_dim, projector_dim, scheduler = None):
    out = os.path.join("./saved_models/", name.format(epoch))

    torch.save({
        "epoch": epoch + 1,
        "encoder_dim": encoder_dim,
        "projector_dim": projector_dim,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler != None else {}
    },out)