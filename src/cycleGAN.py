import torch.nn as nn
import pretrainedmodels as cm


class GANModel(nn.Module):
    def __init__(self, generator, discriminator):
        """
          generator : the generator of the GAN
          discriminator : the discriminator of the GAN
        """
        super(GANModel, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, batch_A, batch_B):
        """
          Forward of the model, compute and return the following
          gen_B_fake = generator(batch_A)
          pred_gen_A = discriminator(gen_B_fake), no grad on discriminator
          pred_disc_A = discriminator(desc_eval_B_fake)
          pred_disc_B =  discriminator(batch_B)

          returns:
            pred_gen_B_fake, pred_B_fake, pred_B_true
        """
        gen_A = self.generator(batch_A)

        pred_disc_A = self.discriminator(gen_A)
        pred_disc_B = self.discriminator(batch_B)

        #discard gradient change on discriminator
        #self.discriminator.eval()
        pred_gen_A = self.discriminator(gen_A)
        #self.discriminator.train()

        return pred_gen_A, pred_disc_A, pred_disc_B

class DiscriminatorModel(nn.Module):
  """
    as models from pretrainedmodels required a 3 channel input
    wrap them in a model that perform a conv that output 3 channels
  """
  def __init__(self, weight = 'imagenet'):
    super(DiscriminatorModel, self).__init__()
    self.layer = nn.Conv2d(1, 3, (2, 2), padding=2)

    model = cm.resnet18(num_classes=1000, pretrained = weight)
    model.last_linear = nn.Linear(in_features=512, out_features=1, bias=True)

    self.model = model

  def forward(self, x):
    x = self.layer(x)
    x = self.model(x)
    return x

class CycleGAN(nn.Module):
    def __init__(self, generator_AB, generator_BA, discriminator_A, discriminator_B):
        """
            generator_AB : the generator to go from CT to T2 images
            generator_BA : the generator to go from T2 to CT images
            discriminator_A : the CT discriminator
            discriminator_B : the T2 discriminator
        """
        super(CycleGAN, self).__init__()
        self.generator_AB = generator_AB
        self.generator_BA = generator_BA
        self.discriminator_A = discriminator_A
        self.discriminator_B = discriminator_B

    def forward(self, batch_A, batch_B):
        """
          batch_A : a batch of real CT images
          batch_B : a batch of real T2 images
        """
        #### STEP 1
        # use generator_AB(batch_A) to generate fake B
        fake_B = self.generator_AB(batch_A)
        # use discriminator_B in eval mode to discriminate B with all fake B
        #self.discriminator_B.eval()
        fake_B_gen = self.discriminator_B(fake_B)
        #self.discriminator_B.train()
        # use(batch_B) in eval mode to go back to A
        #self.generator_BA.eval()
        fake_fake_A = self.generator_BA(fake_B)
        #self.generator_BA.train()

        #### STEP 2
        # use discriminator_B to discriminate B with true and fake B
        real_B_disc = self.discriminator_B(batch_B)
        fake_B_disc = self.discriminator_B(fake_B)

        #### STEP 3
        # use generator_BA(batch_B) to generate fake A
        fake_A = self.generator_BA(batch_B)
        # use discriminator_A in eval mode to discriminate A with all fake A
        #self.discriminator_A.eval()
        fake_A_gen = self.discriminator_A(fake_A)
        #self.discriminator_A.train()
        # use generator_AB(batch_A) in eval mode to go back to B
        #self.generator_AB.eval()
        fake_fake_B = self.generator_AB(fake_A)
        #self.generator_AB.train()

        #### STEP 4
        # use discriminator_A to discriminate A with all true A
        real_A_disc = self.discriminator_A(batch_A)
        fake_A_disc = self.discriminator_A(fake_A)

        return fake_B_gen, fake_fake_A, real_B_disc, fake_B_disc, fake_A_gen, fake_fake_B, real_A_disc, fake_A_disc