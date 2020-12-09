
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pytorch_ssim

from trainer import train_cycle_GAN
from vgg import load_feature_extractor
from cycleGAN import CycleGAN, DiscriminatorModel
from dataset import DomainDataset
from utils import *

NB_EPOCHS = 200
BATCH_SIZE = 2
DEVICE = torch.device('cuda')

if __name__ == '__main__':
    T2_DATA_FOLDER = '/data/T2_slices_registered'
    CT_DATA_FOLDER = '/data/CT_slices_registered_clean'

    # convert PIL image to tensor, otherwise, DataLoader throw error
    convertPIL2Tensor = transforms.ToTensor()
    transformation = transforms.Compose([convertPIL2Tensor])

    # Build the T2 and CT dataset
    T2Dataset = DomainDataset(T2_DATA_FOLDER, transformation)
    CTDataset = DomainDataset(CT_DATA_FOLDER, transformation)

    # perform train/CV/test split
    T2_train_id, T2_other = train_test_split(range(len(T2Dataset)), test_size=0.3, random_state=33)
    T2_valid, T2_test = train_test_split(T2_other, test_size=0.5, random_state=87)
    CT_train_id, CT_other = train_test_split(range(len(CTDataset)), test_size=0.3, random_state=33)
    CT_valid, CT_test = train_test_split(CT_other, test_size=0.5, random_state=87)

    T2Dataset_train = torch.utils.data.Subset(T2Dataset, T2_train_id)
    T2Dataset_valid = torch.utils.data.Subset(T2Dataset, T2_valid)
    T2Dataset_test = torch.utils.data.Subset(T2Dataset, T2_test)

    CTDataset_train = torch.utils.data.Subset(CTDataset, CT_train_id)
    CTDataset_valid = torch.utils.data.Subset(CTDataset, CT_valid)
    CTDataset_test = torch.utils.data.Subset(CTDataset, CT_test)

    print('T2 train/valid/test lengths :', len(T2Dataset_train), len(T2Dataset_valid), len(T2Dataset_test))
    print('CT train/valid/test lengths :', len(CTDataset_train), len(CTDataset_valid), len(CTDataset_test))

    # Create CycleGAN model
    cycleGAN_T2_CT = CycleGAN(getGenerator().to(DEVICE), getGenerator().to(DEVICE),
                              DiscriminatorModel().to(DEVICE), DiscriminatorModel().to(DEVICE))

    generator_CT_opt = torch.optim.Adam(cycleGAN_T2_CT.generator_AB.parameters(), lr=1e-4)
    discriminator_CT_opt = torch.optim.Adam(cycleGAN_T2_CT.discriminator_B.parameters(), lr=1e-3)
    generator_T2_opt = torch.optim.Adam(cycleGAN_T2_CT.generator_BA.parameters(), lr=1e-4)
    discriminator_T2_opt = torch.optim.Adam(cycleGAN_T2_CT.discriminator_A.parameters(), lr=1e-3)

    loss_discriminator = nn.BCEWithLogitsLoss()
    loss_cycle = lambda x, y: -pytorch_ssim.SSIM()(x, y)

    feature_extractors, feature_weights = load_feature_extractor()
    print('Start training')
    train_cycle_GAN(cycleGAN_T2_CT, generator_CT_opt, discriminator_CT_opt, generator_T2_opt,
                    discriminator_T2_opt, loss_discriminator, loss_cycle,
                    T2Dataset_train,
                    CTDataset_train,
                    feature_extractors,
                    feature_weights,
                    T2Dataset_valid,
                    CTDataset_valid,
                    batchsize=BATCH_SIZE, nb_epoch=NB_EPOCHS, valid=5, k_disc=4, k_gen=1)




