import numpy as np
from matplotlib.path import Path
from backend.settings import MEDIA_ROOT
from pydicom import dcmread
import os


def load_data(identifier):
        '''
        Create a 4D matrix (with dimensions x, y, offset, slice) of a previously uploaded 
        dataset and a list of its corresponding offset values.
        '''
 
        root = f'{MEDIA_ROOT}/uploads/{identifier}/images'
        directory = os.listdir(root)
        offsets = []
        data = []
        names = []
        offset_num = -1
        for f in directory:
            ds = dcmread(os.path.join(MEDIA_ROOT, f'uploads/{identifier}/{f[:-4]}.dcm'))
            protocol_name = ds['ProtocolName'].value.split('_')
            ppm_index = protocol_name.index("ppm")
            names += [protocol_name[ppm_index - 1]]
            if int(f[:-4].split('-')[-1]) == 1:
                offset_num += 1
                data += [[ds.pixel_array]]
                try:
                    offsets += [float(protocol_name[ppm_index - 1])]
                except:
                     offsets += [0]
            else:
                data[offset_num].append(ds.pixel_array)
            
        return data, offsets, names


def pointsToMask(poly_verts_list, nx, ny):
        '''Convert a list of polygon vertices to a list of masks

        >>> vertices = [[
                [207.40625, 229.203125],
                [272.40625, 165.203125],
                [345.40625, 238.203125],
                [276.40625, 302.203125],
                [198.40625, 283.203125]
               ]]
        >>> result = pointsToMask(vertices, 500, 500)
        >>> f = open("pointsToMaskExample.txt")
        >>> correct = f.read()
        >>> result == correct
        True
        '''
        
        nx, ny = int(nx), int(ny) # TODO: Figure out how to address small ROI size
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        grid = np.zeros(nx * ny, dtype=bool)

        if len(poly_verts_list[0]) == 2:
            poly_verts_list = [[point[1]] for point in poly_verts_list]

        for poly_verts in poly_verts_list:
        
            scaled_verts = [[v[0] * 0.25, v[1] * 0.25] for v in poly_verts]
            path = Path(scaled_verts)
            # Check if points are inside the polygon
            poly_mask = path.contains_points(points)
            grid = np.logical_or(grid, poly_mask)

        grid = grid.reshape((ny, nx))
        return grid
