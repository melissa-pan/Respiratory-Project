from train import train_net
from train import get_model_name
from model import SoundNet
import torch

resume = True

net = SoundNet()

batch = 256

if resume:
    model_path = get_model_name(net.name, batch, 0.0005, 9)
    net.load_state_dict(torch.load(model_path))
    train_net(net, batch_size=batch, learning_rate=0.0005, num_epochs=20, reload_epoch=10, model_path=model_path)
else:
    train_net(net, batch_size=batch, learning_rate=0.0005, num_epochs=10)
