'''
Run segmentation on the orthomosaic (docker/ODM)
Random resize crop the orthomosaic to get training images
Put markers/identifiers on the cropped images
'''

from HN_image_splitting import photoset, read_orthophotos, read_tiles
import torchvision.transforms as T
from matplotlib import pyplot as plt
from HN_watershed import segmentation, gamma_correction
from HN_image_splitting import make_photosets, make_sets
import cv2 as cv

### FILEPATHS ###
filepaths_2_45 = [
'tile_01_04.png',
'tile_01_05.png',
'tile_01_06.png',
'tile_01_07.png',
'tile_02_01.png',
'tile_02_02.png',
'tile_02_03.png',
'tile_02_04.png',
'tile_02_05.png',
'tile_02_06.png',
'tile_02_07.png',
'tile_03_01.png',
'tile_03_02.png',
'tile_03_03.png',
'tile_03_04.png',
'tile_03_05.png',
'tile_03_06.png',
'tile_03_07.png',
'tile_04_01.png',
'tile_04_02.png',
'tile_04_03.png',
'tile_04_04.png',
'tile_04_05.png',
'tile_04_06.png',
'tile_04_07.png',
'tile_05_01.png',
'tile_05_02.png',
'tile_05_03.png',
'tile_05_04.png',
'tile_05_05.png',
'tile_05_06.png',
'tile_05_07.png',
'tile_06_01.png',
'tile_06_02.png',
'tile_06_03.png',
'tile_06_04.png',
'tile_06_05.png',
'tile_06_06.png',
'tile_06_07.png',
'tile_07_01.png',
'tile_07_02.png',
'tile_07_03.png',
'tile_07_04.png',
'tile_07_05.png',
'tile_07_06.png',
]


filepaths_3_49 = [
'tile_01_01.png',
'tile_01_02.png',
'tile_01_03.png',
'tile_01_04.png',
'tile_01_05.png',
'tile_01_06.png',
'tile_01_07.png',
'tile_02_01.png',
'tile_02_02.png',
'tile_02_03.png',
'tile_02_04.png',
'tile_02_05.png',
'tile_02_06.png',
'tile_02_07.png',
'tile_03_01.png',
'tile_03_02.png',
'tile_03_03.png',
'tile_03_04.png',
'tile_03_05.png',
'tile_03_06.png',
'tile_03_07.png',
'tile_04_01.png',
'tile_04_02.png',
'tile_04_03.png',
'tile_04_04.png',
'tile_04_05.png',
'tile_04_06.png',
'tile_04_07.png',
'tile_05_01.png',
'tile_05_02.png',
'tile_05_03.png',
'tile_05_04.png',
'tile_05_05.png',
'tile_05_06.png',
'tile_05_07.png',
'tile_06_01.png',
'tile_06_02.png',
'tile_06_03.png',
'tile_06_04.png',
'tile_06_05.png',
'tile_06_06.png',
'tile_06_07.png',
'tile_07_01.png',
'tile_07_02.png',
'tile_07_03.png',
'tile_07_04.png',
'tile_07_05.png',
'tile_07_06.png',
'tile_07_07.png',
]


filepaths_4_39 = [
'tile_01_05.png',
'tile_01_06.png',
'tile_01_07.png',
'tile_02_03.png',
'tile_02_04.png',
'tile_02_05.png',
'tile_02_06.png',
'tile_02_07.png',
'tile_03_01.png',
'tile_03_02.png',
'tile_03_03.png',
'tile_03_04.png',
'tile_03_05.png',
'tile_03_06.png',
'tile_03_07.png',
'tile_04_01.png',
'tile_04_02.png',
'tile_04_03.png',
'tile_04_04.png',
'tile_04_05.png',
'tile_04_06.png',
'tile_04_07.png',
'tile_05_01.png',
'tile_05_02.png',
'tile_05_03.png',
'tile_05_04.png',
'tile_05_05.png',
'tile_05_06.png',
'tile_05_07.png',
'tile_06_02.png',
'tile_06_03.png',
'tile_06_04.png',
'tile_06_05.png',
'tile_06_06.png',
'tile_06_07.png',
'tile_07_02.png',
'tile_07_03.png',
'tile_07_04.png',
'tile_07_05.png',
]



def main():
    # training_set, validation_set = make_sets()
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
    dist_transform_factor = 6 # blanket for now
    # orthophotos = read_orthophotos() # dictionary of the format orthophotos[flight{i}_ortho] = {cv image}

    # change filename lists to just the end and transfer with headers from the folder and from hardcoded names
    basepath = 'C:\\Users\\haley\\UROPSP22\\non-mit\\TreeCrownDelineation\\'
    headers = ['2_45', '3_49', '4_39']
    k = 0
    for folder in [filepaths_2_45, filepaths_3_49, filepaths_4_39]:
        cur_header = headers[k]
        for filename in folder:
            str = basepath + cur_header + '\\' + filename
            photo = cv.imread(str)
            cv.imwrite(basepath + 'all_rgb\\' + cur_header + '_' + filename, photo)

            m, img = segmentation(photo, dist_transform_factor)
            m2, img2 = segmentation(photo, dist_transform_factor) # create copy for resetting background values

            max_label = 0
            for row in m: # find iteration max for re-labeling
                row_max = max(row)
                max_label = max(row_max, max_label)

            # how to change each instance of 1 to an individual incrementing instance of a tree?

            for i in range(2, max_label+1):
                m2[m == i] = 0 # set to background
            
            cv.imwrite(basepath + 'all_marker\\' + cur_header + '_' + filename, m2)
            cv.imwrite(basepath + cur_header + '\\_marker\\' + cur_header + '_' + filename, m2)

            # plt.subplot(1, 2, 1),plt.imshow(img)
            # plt.subplot(1, 2, 2),plt.imshow(m2)
            # plt.show()
        k += 1

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

def check():
    dist_transform_factor = 6 # blanket for now

    # change filename lists to just the end and transfer with headers from the folder and from hardcoded names
    basepath = 'C:\\Users\\haley\\UROPSP22\\non-mit\\TreeCrownDelineation\\'
    headers = ['2_45', '3_49', '4_39']
    k = 0

    cur_header = '3_49'
    filename = 'tile_03_04.png'
    str = basepath + cur_header + '\\' + filename
    photo = cv.imread(str)

    m, img = segmentation(photo, dist_transform_factor)
    m2, img2 = segmentation(photo, dist_transform_factor) # create copy for resetting background values

    max_label = 0
    for row in m: # find iteration max for re-labeling
        row_max = max(row)
        max_label = max(row_max, max_label)

    # how to change each instance of 1 to an individual incrementing instance of a tree?

    for i in range(2, max_label+1):
        m2[m == i] = 0 # set to background

    plt.subplot(1, 3, 1),plt.imshow(img)
    plt.subplot(1, 3, 2),plt.imshow(m)
    plt.subplot(1, 3, 3),plt.imshow(m2)
    plt.show()

# main()
check()