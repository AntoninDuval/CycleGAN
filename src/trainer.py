
import torch
from utils import *

DEVICE = torch.device('cuda')

def cyclePerceptualLoss(T2, T2_reconstructed, CT, CT_reconstructed, feature_extractors, feature_weights):
    T2 = torch.cat((T2, T2, T2), 1).cuda()
    T2_reconstructed = torch.cat((T2_reconstructed, T2_reconstructed, T2_reconstructed), 1).cuda()
    CT = torch.cat((CT, CT, CT), 1).cuda()
    CT_reconstructed = torch.cat((CT_reconstructed, CT_reconstructed, CT_reconstructed), 1).cuda()
    loss = torch.zeros((1, 1)).cuda()
    for extractor, weight in zip(feature_extractors, feature_weights):
        loss += weight * (torch.mean(torch.abs(extractor(T2) - extractor(T2_reconstructed)))
                          + torch.mean(torch.abs(extractor(CT) - extractor(CT_reconstructed))))
    return loss.cuda()

def train_cycle_GAN(model, generator_CT_opt, discriminator_CT_opt, generator_T2_opt, discriminator_T2_opt, loss,
                    loss_cycle,
                    T2_dataset_train, CT_dataset_train, cpl_feature_extractors, cpl_feature_weights,
                    T2_dataset_valid=None, CT_dataset_valid=None,
                    batchsize=4, nb_epoch=2, valid=5, k_disc=1, k_gen=1, plot=True, ):
    """
      train the GAN model with required input
      params:
        model : the model to train
        optimizer_gen : the optimizer for the genenerator
        optimizer_disc : the optimizer for the discriminator
        loss : the loss applied for the discriminator
        T2_dataset_train : the domain T2 dataset for training
        CT_dataset_train : the domain CT dataset for training
        T2_dataset_valid : the domain T2 dataset for validation
        CT_dataset_valid : the domain T2 dataset for validation
        batchsize : batch size for the T2 dataset, CT dataset is inferred
        epoch : number of epoch
        valid : perform validation every x epochs, None if no validation
        k : number of iteration on generator before training the discriminator
        plot : plot a sample on valid iteration

      returns : train /valid errors for generators and discriminators
    """

    # build the loader
    T2Loader_train = torch.utils.data.DataLoader(T2_dataset_train, batch_size=batchsize, shuffle=True)
    CTLoader_train = torch.utils.data.DataLoader(CT_dataset_train,
                                                 batch_size=int(np.floor(
                                                     batchsize * len(CT_dataset_train) / len(T2_dataset_train))),
                                                 shuffle=True)
    if not (T2_dataset_valid is None or CT_dataset_valid is None):
        T2Loader_valid = torch.utils.data.DataLoader(T2_dataset_valid, batch_size=batchsize, shuffle=True)
        CTLoader_valid = torch.utils.data.DataLoader(CT_dataset_valid,
                                                     batch_size=int(np.floor(
                                                         batchsize * len(CT_dataset_valid) / len(T2_dataset_valid))),
                                                     shuffle=True)

    batches_len = min(len(T2Loader_train), len(CTLoader_train))

    # start training loop
    for epoch in range(1, nb_epoch + 1):
        losses_gen_T2 = 0
        losses_disc_CT = 0
        losses_gen_CT = 0
        losses_disc_T2 = 0
        losses_cyc_T2 = 0
        losses_cyc_CT = 0

        valid_losses_gen_T2 = 0
        valid_losses_disc_CT = 0
        valid_losses_gen_CT = 0
        valid_losses_disc_T2 = 0
        valid_losses_cyc_T2 = 0
        valid_losses_cyc_CT = 0

        # count the number of CT batch performed as it depends of k
        batches_CT_disc = 0
        batches_T2_disc = 0
        batches_CT_gen = 0
        batches_T2_gen = 0


        t_batches = zip(T2Loader_train, CTLoader_train)
        for batch_count, (real_T2, real_CT) in enumerate(t_batches):
            model.train()

            # reset all losses
            discriminator_CT_loss = 0
            discriminator_T2_loss = 0
            generator_T2_cycle_consistency_loss = 0
            generator_CT_discriminator_loss = 0
            generator_CT_cycle_consistency_loss = 0
            generator_T2_discriminator_loss = 0

            real_T2 = real_T2.to(DEVICE).float()
            real_CT = real_CT.to(DEVICE).float()

            # one forward pass through the CycleGAN
            fake_CT_gen, fake_fake_T2, real_CT_disc, fake_CT_disc, fake_T2_gen, fake_fake_CT, real_T2_disc, fake_T2_disc = cycleGAN_T2_CT(
                real_T2, real_CT)

            # backward pass for generator_B
            if np.random.choice(range(k_gen)) == 0:
                batches_CT_gen += 1
                generator_T2_cycle_consistency_loss = loss_cycle(real_T2, fake_fake_T2)
                generator_CT_discriminator_loss = loss(fake_CT_gen, torch.ones(fake_CT_gen.shape).to(DEVICE)) \
                                                  + cyclePerceptualLoss(real_T2, fake_fake_T2, real_CT, fake_fake_CT,
                                                                        cpl_feature_extractors, cpl_feature_weights)

                losses_gen_CT += generator_CT_discriminator_loss.item()
                losses_cyc_T2 += generator_T2_cycle_consistency_loss.item()
                t_batches.set_postfix(G_CT=safe_div(losses_gen_CT, batches_CT_gen),
                                      D_CT=safe_div(losses_disc_CT, batches_CT_disc),
                                      G_T2=safe_div(losses_gen_T2, batches_T2_gen),
                                      D_T2=safe_div(losses_disc_T2, batches_T2_disc),
                                      C_T2=safe_div(losses_cyc_T2, batches_CT_gen),
                                      C_CT=safe_div(losses_cyc_CT, batches_T2_gen))

            # backward pass for discriminator B
            if np.random.choice(range(k_disc)) == 0:
                batches_CT_disc += 1

                discriminator_CT_loss = (loss(real_CT_disc, torch.ones(real_CT_disc.shape).to(DEVICE)) \
                                         + loss(fake_CT_disc, torch.zeros(fake_CT_disc.shape).to(DEVICE))) / 2

                losses_disc_CT += discriminator_CT_loss.item()
                t_batches.set_postfix(G_CT=safe_div(losses_gen_CT, batches_CT_gen),
                                      D_CT=safe_div(losses_disc_CT, batches_CT_disc),
                                      G_T2=safe_div(losses_gen_T2, batches_T2_gen),
                                      D_T2=safe_div(losses_disc_T2, batches_T2_disc),
                                      C_T2=safe_div(losses_cyc_T2, batches_CT_gen),
                                      C_CT=safe_div(losses_cyc_CT, batches_T2_gen))

            # backward pass for generator_BA
            if np.random.choice(range(k_gen)) == 0:
                batches_T2_gen += 1
                generator_CT_cycle_consistency_loss = loss_cycle(real_CT, fake_fake_CT)
                generator_T2_discriminator_loss = loss(fake_T2_gen, torch.ones(fake_T2_gen.shape).to(DEVICE)) \
                                                  + cyclePerceptualLoss(real_T2, fake_fake_T2, real_CT, fake_fake_CT,
                                                                        cpl_feature_extractors, cpl_feature_weights)

                losses_gen_T2 += generator_T2_discriminator_loss.item()
                losses_cyc_CT += generator_CT_cycle_consistency_loss.item()
                t_batches.set_postfix(G_CT=safe_div(losses_gen_CT, batches_CT_gen),
                                      D_CT=safe_div(losses_disc_CT, batches_CT_disc),
                                      G_T2=safe_div(losses_gen_T2, batches_T2_gen),
                                      D_T2=safe_div(losses_disc_T2, batches_T2_disc),
                                      C_T2=safe_div(losses_cyc_T2, batches_CT_gen),
                                      C_CT=safe_div(losses_cyc_CT, batches_T2_gen))

            # backward pass for discriminator A
            if np.random.choice(range(k_disc)) == 0:
                batches_T2_disc += 1
                discriminator_T2_loss = (loss(real_T2_disc, torch.ones(real_T2_disc.shape).to(DEVICE)) \
                                         + loss(fake_T2_disc, torch.zeros(fake_T2_disc.shape).to(DEVICE))) / 2

                losses_disc_T2 += discriminator_T2_loss.item()
                t_batches.set_postfix(G_CT=safe_div(losses_gen_CT, batches_CT_gen),
                                      D_CT=safe_div(losses_disc_CT, batches_CT_disc),
                                      G_T2=safe_div(losses_gen_T2, batches_T2_gen),
                                      D_T2=safe_div(losses_disc_T2, batches_T2_disc),
                                      C_T2=safe_div(losses_cyc_T2, batches_CT_gen),
                                      C_CT=safe_div(losses_cyc_CT, batches_T2_gen))

            # generator CT gradient update
            generator_CT_opt.zero_grad()
            generator_CT_global_loss = 0.5 * generator_T2_cycle_consistency_loss + 1 * generator_CT_discriminator_loss
            if not type(generator_CT_global_loss) == int:
                generator_CT_global_loss.backward(retain_graph=True)
                generator_CT_opt.step()

            # generator T2 gradient update
            generator_T2_opt.zero_grad()
            generator_T2_global_loss = 0.5 * generator_CT_cycle_consistency_loss + 1 * generator_T2_discriminator_loss
            if not type(generator_T2_global_loss) == int:
                generator_T2_global_loss.backward(retain_graph=True)
                generator_T2_opt.step()

            # discriminator CT gradient update
            discriminator_CT_opt.zero_grad()
            discriminator_CT_global_loss = 1 * discriminator_CT_loss
            if not type(discriminator_CT_global_loss) == int:
                discriminator_CT_global_loss.backward()
                discriminator_CT_opt.step()

            # discriminator T2 gradient update
            discriminator_T2_opt.zero_grad()
            discriminator_T2_global_loss = 1 * discriminator_T2_loss
            if not type(discriminator_T2_global_loss) == int:
                discriminator_T2_global_loss.backward()
                discriminator_T2_opt.step()

        # validation part
        if not (valid is None or T2_dataset_valid is None or CT_dataset_valid is None):
            if epoch % valid == 0:
                _ = valid_cycle_GAN(model, loss, loss_cycle, T2Loader_valid, CTLoader_valid, epoch)

        # save every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save(cycleGAN_T2_CT.state_dict(), f'data/model/full_cycle_ep{epoch + 1}.h5')

    return 0


def valid_cycle_GAN(model, loss, cycle_loss, T2_loader, CT_loader, epoch):
    """
        validation loop for gan model
        params:
          model : the model to be evaluated
          loss : the loss the evaluation will refer to
          T2_loader : data loader for the T2 domain
          CT_loader : the loader for the CT domain
        returns:
          generator and descriminator loss
    """
    model.eval()
    with torch.no_grad():
        losses_gen_T2 = 0
        losses_disc_CT = 0
        losses_gen_CT = 0
        losses_disc_T2 = 0
        losses_cyc_T2 = 0
        losses_cyc_CT = 0

        batches_len = min(len(T2_loader), len(CT_loader))

        # init tqdm bar
        t_batches = tqdm(zip(T2_loader, CT_loader), f'epoch {epoch} validation', total=batches_len)

        # loop over valid set
        for batch_count, (T2_real, CT_real) in enumerate(t_batches):
            T2_real = T2_real.to(DEVICE).float()
            CT_real = CT_real.to(DEVICE).float()

            # intialize labels for discriminator
            CT_ones = torch.Tensor(np.ones((len(CT_real), 1))).to(DEVICE)
            CT_zeros = torch.Tensor(np.zeros((len(CT_real), 1))).to(DEVICE)
            T2_ones = torch.Tensor(np.ones((len(T2_real), 1))).to(DEVICE)
            T2_zeros = torch.Tensor(np.zeros((len(T2_real), 1))).to(DEVICE)

            fake_CT_gen, fake_fake_T2, real_CT_disc, fake_CT_disc, fake_T2_gen, fake_fake_CT, real_T2_disc, fake_T2_disc = cycleGAN_T2_CT(
                T2_real, CT_real)

            losses_gen_T2 += loss(fake_CT_gen, T2_ones).item()
            losses_disc_CT += (loss(real_CT_disc, CT_ones) + loss(fake_CT_gen, T2_zeros)).item() * 0.5
            losses_gen_CT += loss(fake_T2_gen, CT_ones).item()
            losses_disc_T2 += (loss(real_T2_disc, T2_ones) + loss(fake_T2_disc, CT_zeros)).item() * 0.5
            losses_cyc_T2 += loss_cycle(T2_real, fake_fake_T2).item()
            losses_cyc_CT += loss_cycle(CT_real, fake_fake_CT).item()

            t_batches.set_postfix(G_CT=safe_div(losses_gen_CT, batch_count), D_CT=safe_div(losses_disc_CT, batch_count),
                                  G_T2=safe_div(losses_gen_T2, batch_count), D_T2=safe_div(losses_disc_T2, batch_count),
                                  C_T2=safe_div(losses_cyc_T2, batch_count), C_CT=safe_div(losses_cyc_CT, batch_count))

    return 0


def plot_sample_cycle_GAN(model, T2_dataset, CT_dataset, id_):
    """
      plot a sample image
      params:
        model : the model to plot a sample with
        T2_dataset : the dataset we want to use
        id_ : the image id in loader we want to use
    """
    model.eval()
    with torch.no_grad():
        fake_CT = model.generator_AB(torch.tensor(T2_dataset[id_]).unsqueeze(0).float().to(DEVICE))
        fake_fake_T2 = model.generator_BA(fake_CT).unsqueeze(0).float().to(DEVICE)
        fake_T2 = model.generator_BA(torch.tensor(CT_dataset[id_]).unsqueeze(0).float().to(DEVICE))
        fake_fake_CT = model.generator_AB(fake_T2).unsqueeze(0).float().to(DEVICE)

    plt.figure()
    _, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax[0][0].imshow(T2_dataset[id_].squeeze().numpy(), cmap='gray')
    ax[0][1].imshow(fake_CT.squeeze().cpu().numpy(), cmap='gray')
    ax[0][2].imshow(fake_fake_T2.squeeze().cpu().numpy(), cmap='gray')
    ax[1][0].imshow(CT_dataset[id_].squeeze().numpy(), cmap='gray')
    ax[1][1].imshow(fake_T2.squeeze().cpu().numpy(), cmap='gray')
    ax[1][2].imshow(fake_fake_CT.squeeze().cpu().numpy(), cmap='gray')

    _ = plt.show()