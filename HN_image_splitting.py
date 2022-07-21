import image_slicer
from PIL import Image # change requirements.txt on import to remove Pillow version, 
from HN_watershed import gamma_correction # personal implementation
from HN_watershed import segmentation # personal implementation

from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import os
import random

# # Read in each orthophoto
def read_orthophotos():
    orthophotos = dict()
    for i in range(2, 5):
        orthophotos[f'flight{i}_ortho'] = cv.imread(rf'C:\Users\haley\UROPSP22\Clean\test_orthophoto\flight{i}_orthophoto.TIF')
    return orthophotos

# # Read in each tile
def read_tiles(flight, split_amount, trim=True):
    images = []
    folder = rf'C:\Users\haley\UROPSP22\Clean\photosets_{split_amount}\{flight}'
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None: 
            images.append(img)
    return images

# # Display corrected and original images (for debugging):
# corrected = gamma_correction(orthophotos['flight2_ortho'])
# plt.subplot(1, 2, 1),plt.imshow(corrected)
# plt.subplot(1, 2, 2),plt.imshow(orthophotos['flight2_ortho']) # show original photo
# plt.show()

# # Gamma correct all orthophotos (could move in-line above, separate for debgugging and comparison)
def gamma_correct_all(orthophotos):
    corrected_ortho = dict()
    for key, photo in orthophotos.items():
        corrected_ortho[key] = gamma_correction(photo)
    return corrected_ortho

# # Photoset class and function for containing and observing sets of sliced photos from orthophotos
split_amount = 48
class photoset():
    def __init__(self, orthophoto, flight_num):
        self.flight_num = flight_num
        self.photo = orthophoto
        self.total_height, self.total_width, self.color = orthophoto.shape
        self.tiles = image_slicer.slice(rf'C:\Users\haley\UROPSP22\Clean\test_orthophoto\flight{flight_num}_orthophoto.TIF', split_amount, save=False)
        self.split_images = [tile for tile in self.tiles] #listify each tile as a PIL image

def make_photosets(orthophotos):
    photosets = dict()
    for i, photo in enumerate(orthophotos.values()):
        photosets[i+2] = photoset(photo, i+2) # +2 indicates first real flight number (starts at 2)
    return photosets

def get_orthophotos(flight, photoset_num, dtf):
    # # Read in tiles from flight 2, 48 photoset
    # seg_params = {2: }
    # dist_transform_factor = 6 # blanket for now
    markers = []
    segmented = []
    flight2_tiles = read_tiles(flight, photoset_num, trim=True)
    for tile in flight2_tiles:
        m, img = segmentation(tile, dtf)
        markers.append(m)
        segmented.append(gamma_correction(img))
    
    return markers, segmented

def make_sets():
    dtf = 6 # blanket for now
    pn = 48 # photoset number
    markers2, segmented2 = get_orthophotos(2, pn, dtf)
    markers3, segmented3 = get_orthophotos(3, pn, dtf)
    markers4, segmented4 = get_orthophotos(4, pn, dtf)

    training_set_imgs = [] # 70% = 31, 34, 27
    training_set_markers = []
    validation_set_imgs = [] # 30% = 14, 15, 12
    validation_set_markers = []

    set_70 = {2: 31, 3: 34, 4: 27}
    set_30 = {2: 14, 3: 15, 4: 12}

    def split_flight(flight_num):
        t_set_imgs = []
        t_set_markers = []
        v_set_imgs = []
        v_set_markers = []

        t_set_markers.extend(random.choices(f'markers{flight_num}', k=set_70[flight_num]))
        indices = [f'markers{flight_num}'.index(tile) for tile in t_set_markers]
        for i in range(len(f'markers{flight_num}')):
            if i not in indices:
                v_set_markers.append(f'markers{flight_num}'[i])
                v_set_imgs.append(f'segmented{flight_num}'[i])
            else:
                t_set_imgs.append(f'segmented{flight_num}'[i])

        return t_set_imgs, t_set_markers, v_set_imgs, v_set_markers
    
    t2_i, t2_m, v2_i, v2_m = split_flight(2)
    t3_i, t3_m, v3_i, v3_m = split_flight(3)
    t4_i, t4_m, v4_i, v4_m = split_flight(4)

    training_set_imgs.extend(t2_i)
    training_set_imgs.extend(t3_i)
    training_set_imgs.extend(t4_i)

    training_set_markers.extend(t2_m)
    training_set_markers.extend(t3_m)
    training_set_markers.extend(t4_m)

    validation_set_imgs.extend(v2_i)
    validation_set_imgs.extend(v3_i)
    validation_set_imgs.extend(v4_i)

    validation_set_markers.extend(v2_m)
    validation_set_markers.extend(v3_m)
    validation_set_markers.extend(v4_m)

    return training_set_imgs, training_set_markers, validation_set_imgs, validation_set_markers
    
# # Make photosets and save photoset tiles to the directory 
def main():
    # # Make training and validation sets
    pass

    # # Make Photosets
    # orthophotos = read_orthophotos()
    # corrected_ortho = gamma_correct_all(orthophotos)
    # photosets = make_photosets(corrected_ortho)

    # # Save tiles
    # for i, p_set in enumerate(photosets.values()):
    #     image_slicer.save_tiles(p_set.tiles, directory=rf'C:\Users\haley\UROPSP22\Clean\photosets_{split_amount}\{i+2}', prefix='tile', format='png')

    # # Segment tile images for comparison to orthomosaic segmentation -- not finished
    # for i, p_set in enumerate(photosets.values()):
    #     image_slicer.save_tiles(p_set.tiles, directory=rf'C:\Users\haley\UROPSP22\Clean\photosets_{split_amount}\{i+2}', prefix='tile', format='png')  


main()