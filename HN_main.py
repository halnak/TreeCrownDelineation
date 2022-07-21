'''
Run segmentation on the orthomosaic (docker/ODM)
Random resize crop the orthomosaic to get training images
Put markers/identifiers on the cropped images
'''

from HN_image_splitting import read_orthophotos, read_tiles
import torchvision.transforms as T
from matplotlib import pyplot as plt
from HN_watershed import segmentation, gamma_correction
from HN_image_splitting import make_photosets, make_sets

def main():
    training_set, validation_set = make_sets()
    # orthophotos = read_orthophotos() # dictionary of the format orthophotos[flight{i}_ortho] = {cv image}

    # # Read in tiles from flight 2, 48 photoset
    # # seg_params = {2: }
    # dist_transform_factor = 6 # blanket for now
    # markers = []
    # segmented = []
    # flight2_tiles = read_tiles(2, 48, trim=True)
    # for tile in flight2_tiles:
    #     m, img = segmentation(tile, dist_transform_factor)
    #     markers.append(m)
    #     segmented.append(gamma_correction(img))

    # # Display the above
    # plt.subplot(1, 4, 1),plt.imshow(segmented[10])
    # plt.subplot(1, 4, 2),plt.imshow(segmented[11])
    # plt.subplot(1, 4, 3),plt.imshow(segmented[12])
    # plt.subplot(1, 4, 4),plt.imshow(segmented[13])
    # plt.show()


    # # Segment and display images
    # seg_params = {2: }
    # dist_transform_factor = 6 # blanket for now
    # orthophotos = read_orthophotos() # dictionary of the format orthophotos[flight{i}_ortho] = {cv image}
    # markers = []
    # segmented = []
    # for i, photo in enumerate(orthophotos.values()):
    #     m, img = segmentation(photo, dist_transform_factor)
    #     markers.append(m)
    #     segmented.append(gamma_correction(img))
    
    # plt.subplot(1, 3, 1),plt.imshow(segmented[0])
    # plt.subplot(1, 3, 2),plt.imshow(segmented[1])
    # plt.subplot(1, 3, 3),plt.imshow(segmented[2])
    # plt.show()


    # # Random resize crop
    # orthophotos = read_orthophotos() # dictionary of the format orthophotos[flight{i}_ortho] = {cv image}
    # randomresize = []
    # for photo in orthophotos.values():
    #     height, width, color = photo.shape
    #     output = T.RandomResizedCrop(size=(256, 256), scale=(224/height, 224/width),ratio=(0.75, 1.3333333333333333), interpolation=T.InterpolationMode.BILINEAR) # all other params on default, 256 and 224 provided by Onishi et al
    #     randomresize.append(output)
    
    # plt.subplot(1, 3, 1),plt.imshow(randomresize[0])
    # plt.subplot(1, 3, 2),plt.imshow(randomresize[1])
    # plt.subplot(1, 3, 3),plt.imshow(randomresize[2])
    # plt.show()

main()