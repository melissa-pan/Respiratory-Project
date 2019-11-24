from train import get_model_name
from train import plot_training_curve

path = get_model_name('Sound', 256, 0.0005, 19)
plot_training_curve(path)