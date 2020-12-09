import SimpleITK as sitk
import png
import numpy as np
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import random
import matplotlib.pyplot as plt

def safe_div(x,y):
    """ return 0 is denominator is 0 """
    return x/y if y else 0

def full_image_to_slice(path, slice):
    '''
    From a 3D image, save a slice corresponding to the index. Normalize the pixel image.
    Return the minimum size of the image.
    :param path:
    :param slice:
    :return:
    '''
    items = [f for f in listdir(path) if isfile(join(path, f))]
    min_size_x, min_size_y = [], []
    for name in items:
      new_name = name[:-7] + '.png'
      image = sitk.ReadImage(path + "/" + name)
      image_np = sitk.GetArrayFromImage(image)
      if(path == 'T1'):
        if image_np.shape[1] < slice:
          continue
        image_np = image_np[:, slice, :].astype(np.int16)
      else:
        if image_np.shape[0] < slice:
          continue
        image_float_and_flip = np.flip(image_np[slice, :, :].astype(np.float).T).copy()
        # rescale image to 0-255 range
        image_np = (image_float_and_flip / image_float_and_flip.max() * 255).astype(np.uint8)
      min_size_x.append(image_np.shape[0])
      min_size_y.append(image_np.shape[1])
      png.from_array(image_np, 'L').save(path + "_slices/" + new_name)
    return min_size_x, min_size_y

def visualize_random_images(path, items, n=9):
  fig, m_axs = plt.subplots(3, 3, figsize = (20, 20))
  for c_ax, c_row in zip(m_axs.flatten(), random.sample(items,n)):
        c_img = sitk.ReadImage(path + "/" + c_row)
        c_array = sitk.GetArrayFromImage(c_img)

        index = random.randint(0,2)
        coord = random.randint(0,c_array.shape[0])

        x_coord = ':'
        y_coord = ':'
        z_coord = ':'

        if index==0:
          c_slice = c_array[coord,:,:]
          x_coord = coord
        elif index==1:
          c_slice = c_array[:,coord,:]
          y_coord = coord
        elif index==2:
          c_slice = c_array[:,:,coord]
          z_coord = coord

        c_ax.imshow(c_slice, cmap='bone')
        c_ax.set_title('Image shape : {} / Coordinate : ({},{},{})'.format(c_array.shape,x_coord,y_coord,z_coord))


def modify_pixel_values(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < 0:
                img[i][j] = 0
            if img[i][j] > 255:
                img[i][j] = 255

    return img


def crop_image(c_slice):
    im = Image.fromarray(c_slice).convert('RGB')
    width, height = im.size  # Get dimensions

    left = (width - 300) / 2
    top = (height - 300) / 2
    right = (width + 300) / 2
    bottom = (height + 300) / 2

    # Crop the center of the image
    im2 = im.crop((left, top, right, bottom))
    return im2

def percent_black_pixel(img):
    """
    Return the % of black pixel in a slice. Information can indicate if the image is of good quality.
    :param img:
    :return:
    """
    number_outter_pixels = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] < 0:
                number_outter_pixels += 1
    number_outter_pixels = (number_outter_pixels / (img.shape[0] * img.shape[1])) * 100

    return number_outter_pixels

def getGenerator(baseline = 'resnet18', weight = 'imagenet', batchnorm = True, activation = nn.ReLU):
  """
    todo
  """
  return sm.Unet(baseline, encoder_weights = weight, decoder_use_batchnorm = batchnorm, activation = activation, in_channels=1)

def getDiscriminator(weight = 'imagenet'):
  """
    todo
  """
  model = cm.resnet18(num_classes=1000, pretrained = weight)
  model.last_linear = nn.Linear(in_features=512, out_features=1, bias=True)
  return model