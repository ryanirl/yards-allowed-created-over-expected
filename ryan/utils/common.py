import torch


def save_model(filename, model):
    torch.save(model.state_dict(), filename)


def load_model(filename, model):
    model.load_state_dict(torch.load(filename))
    return model



