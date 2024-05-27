import matplotlib.pyplot as plt
import numpy as np
import math
from statistics import mean
from PIL import Image
from scipy.optimize import least_squares

def create_spectrum_roi(data, epi, endo):
    roi_list = ["epi", "endo"]
    rois = [epi.astype(int), endo.astype(int)]
    masks = np.zeros((data.shape + (len(roi_list),)))
    map_mask = np.zeros((data.shape + (len(roi_list),)))
    
    for i in range(np.size(data, 2)):
        dataslice = data[:,:,:,i]
        for j, mask in enumerate(rois):
            mask_spectrum = np.stack([mask]*np.size(data, 2), -1)
            ##Apply masks to spectral data##
            data_mask = dataslice*mask_spectrum
            ##Store masks in array##
            masks[:,:,:,i,j] = data_mask
            map_mask[:,:,i,j] = mask
    return masks, map_mask


def segment(image, mask, arv, irv):
    '''
    Takes an image and a mask of the myocardium (ones inside) and returns a
    cell array where the contents of each element represent the pixels that
    are within that AHA segment. The result can then be used to compute
    means, standard deviations etc. for each segment.

    RETURNS: (segmented_pixels, segmented_indices), where
        segmented_pixels  =  An array where each element is the pixels from
                             the image within each sector
        segmented_indices =  An array where each element is the indices of
                             the mask within each sector
    '''
    
    def centroid(array):
        x_c = 0
        y_c = 0
        area = array.sum()
        it = np.nditer(array, flags=['multi_index'])
        for i in it:
            x_c = i * it.multi_index[1] + x_c
            y_c = i * it.multi_index[0] + y_c
        return (int(x_c/area), int(y_c/area))
    
    # Find centroid and mask coordinates
    [cx, cy] = centroid(mask)
    [y, x] = np.nonzero(mask)
    inds = np.nonzero(mask)
    inds = list(zip(inds[0], inds[1]))

    # Offset all points by centroid
    x = x - cx
    y = y - cy
    arvx = arv[0] - cx
    arvy = arv[1] - cy
    irvx = irv[0] - cx
    irvy = irv[1] - cy

    # Find angular segment cutoffs
    pi = math.pi
    angle = lambda a, b : (math.atan2(a, b)) % (2*pi)
    arv_ang = angle(arvy, arvx)
    irv_ang = angle(irvy, irvx)
    ang = [angle(yc, xc) for yc, xc in zip(y, x)]
    sept_cutoffs = np.linspace(0, arv_ang - irv_ang, num=3) # two septal segments
    wall_cutoffs = np.linspace(arv_ang - irv_ang, 2*pi, num=5)  # four wall segments
    cutoffs = []
    cutoffs.extend(sept_cutoffs)
    cutoffs.extend(wall_cutoffs[1:])
    ang = [(a - irv_ang) % (2*pi) for a in ang]

    # Create arrays of each pixel/index in each segment
    segment_image = lambda a, b : [j for (i,j) in enumerate(inds) if ang[i] >= a and ang[i] < b]
    get_pixels = lambda inds : [image[0][i] for i in inds]

    segmented_indices = [segment_image(a, b) for a, b in zip(cutoffs[:6], cutoffs[1:])]
    segmented_pixels = [get_pixels(inds) for inds in segmented_indices]

    return (segmented_pixels, segmented_indices)


def generate_zspec(images, masks, arvs, irvs, ref):

    signal_intensities = [[] for j in range(len(images))]
    signal_mean = [[] for j in range(len(images))]
    signal_std = [[] for j in range(len(images))]
    signal_n = [[] for j in range(len(images))]
    indices = []
    values = []

    for i in range(len(images)):
        intensities, inds = segment(images[i], masks[i], arvs[i], irvs[i])
        values.append(intensities)
        indices.append(inds)

        if i == ref:
            for seg in range(6):
                v = np.array(intensities[seg])
                ids = np.isfinite(v)
                ref_mean = mean(v[ids])
        else:
            for seg in range(6):
                v = np.array(intensities[seg])
                ids = np.isfinite(v)
                signal_intensities[i].append(v[ids])
                signal_mean[i].append(mean(v[ids]))
                signal_std[i].append(np.std(v[ids]))
                signal_n[i].append(len(v[ids]))
    
    intensities, inds = segment(images[i], masks[i], arvs[i], irvs[i])

    zspec = []
    signal_mean =  [m for m in signal_mean if m != []]
    for seg in range(6):
        spectrum = [m/ref_mean for m in np.transpose(signal_mean)[seg]]
        zspec.append(spectrum)

    return zspec, signal_mean, signal_std, signal_n, signal_intensities, indices


def show_segmentation(mask, segmented_indices):
    segmented = np.zeros((np.size(mask, 0), np.size(mask, 1), 3))
    coords0 = segmented_indices[0] # (255, 0, 0)     red
    coords1 = segmented_indices[1] # (0, 255, 0)     green
    coords2 = segmented_indices[2] # (0, 0, 255)     blue
    coords3 = segmented_indices[3] # (255, 165, 0)   orange
    coords4 = segmented_indices[4] # (255, 255, 100) yellow
    coords5 = segmented_indices[5] # (128, 0, 128)   purple
    
    for i in range(np.size(mask, 0)):
        for j in range(np.size(mask, 1)):
            if (i, j) in coords0:
                segmented[i][j] = np.array([255, 0, 0], dtype=np.uint8())
            elif (i, j) in coords1:
                segmented[i][j] = np.array([0, 255, 0], dtype=np.uint8())
            elif (i, j) in coords2:
                segmented[i][j] = np.array([0, 0, 255], dtype=np.uint8())
            elif (i, j) in coords3:
                segmented[i][j] = np.array([255, 165, 0], dtype=np.uint8())
            elif (i, j) in coords4:
                segmented[i][j] = np.array([255, 255, 100], dtype=np.uint8())
            elif (i, j) in coords5:
                segmented[i][j] = np.array([128, 0, 128], dtype=np.uint8())
            elif mask[i][j] == 1:
                segmented[i][j] = np.array([255, 255, 255], dtype=np.uint8())
            else:
                segmented[i][j] = np.array([0, 0, 0], dtype=np.uint8())

    im = Image.fromarray(segmented.astype(np.uint8))
    im.save("mask.png")


def b0_correction(x, zspec):
    '''Calculate the B0 correction for an input offset list and zspectra.
       Returns the B0 shift for each image and a corrected offset list.
    '''

    #       M0 |~Water~~~~~~~~~~~~~~~~~~~~~|      |~MT~~~~~~~~~~~~~~~~~~~~~~~~|
    #          | Amplitude | FWHM | Center |      | Amplitude | FWHM | Center |
    P0 = [1,    0.8,         1.8,    0,            0.15,        40,     -1     ]
    lb = [0.9,  0.02,        0.3,    -10,          0.0,         30,     -2.5   ]
    ub = [1,    1,           10,     10,           0.5,         60,     0      ]
    
    b0_shift = []
    correct_offsets = []

    for seg in range(6):
        example = lambda p1, p2, p3, x : np.divide(p1 * pow(p2, 2), np.add(pow(p2, 2), 4*(np.power(np.subtract(x, p3), 2))))
        peak = lambda p1, p2, p3 : np.divide(p1 * pow(p2, 2), np.add(pow(p2, 2), 4*(np.power(np.subtract(x, p3), 2))))
        fun = lambda P : np.subtract(P[0], np.add(peak(P[1], P[2], P[3]), peak(P[3], P[4], P[5])))
        resids = lambda P : np.power(np.subtract(fun(P), zspec[seg]), 2)

        T = least_squares(resids, P0, bounds=(lb, ub))
        b0_shift.append(T['x'][3])
        correct_offsets.append([initial - T['x'][3] for initial in x])

        ints = example(0.8, 1.8, 0, list(np.linspace(-10,10,num=100)))
        if seg == 0:
            print([{'x': x, 'y': 1-y} for (x,y) in zip(list(np.linspace(-10,10,num=100)), ints)])

    return (correct_offsets, b0_shift)


def lorentzian_fit(freqs, zspec):
    print()


def show_lorentzian(freqs, zspec):
    fig, ax = plt.subplots()
    ax.plot(freqs, zspec, linewidth=2.0)
    plt.show()