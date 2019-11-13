from train import train_net
from model import SoundNet

net = SoundNet()
train_net(net, batch_size=512, learning_rate=0.001, num_epochs=5)