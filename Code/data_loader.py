import matplotlib.pyplot as plt
import torch

import torchaudio
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import DatasetFolder

#waveform, sample_rate = torchaudio.load('/Users/Ivy/Github/Respiratory-Project/kaggle_db/processed_sound/train/COPD/COPD_120_4742.wav')
# t = torchaudio.transforms.MFCC(sample_rate)(waveform)
# #tn = torchvision.transforms.Normalize(0.5)(t)
# plt.figure()
# plt.imshow(t[0,:,:].detach().numpy())
# print(" ")

# train_iter = torchtext.data.BucketIterator(trainset,
#                                            batch_size=32,
#                                            sort_key=lambda x: len(x), # to minimize padding
#                                            sort_within_batch=True,        # sort within each batch
#                                            repeat=False)                  # repeat the iterator for many epochs


# for b in train_iter:
#     plt.figure()
#     plt.imshow(b[0][1].squeeze().detach().numpy())
#     plt.show()
#     break;

#helper function
def pad(x):
    padded = torch.zeros([40, 1727])
    padded[:, :x.size()[2]] = x.squeeze()
    return padded

def get_data_loader(batch_size):
    tsfm = torchvision.transforms.Compose([
        lambda x: torchaudio.transforms.MFCC(x[1])(x[0]),  # MFCC
        lambda x: pad(x)])

    trainset = DatasetFolder(root='../kaggle_db/processed_sound/train/', loader=torchaudio.load,
                             extensions='.wav', transform=tsfm)

    valset = DatasetFolder(root='../kaggle_db/processed_sound/validation/', loader=torchaudio.load,
                             extensions='.wav', transform=tsfm)

    testset = DatasetFolder(root='../kaggle_db/processed_sound/test/', loader=torchaudio.load,
                             extensions='.wav', transform=tsfm)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, val_loader, None, trainset.class_to_idx
