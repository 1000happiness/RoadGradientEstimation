import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

def reprojection_to_3D(depth_img, camera_mat):
    '''convert depth_img(X * Y * 1) to 3d_img(X * Y * 3) and 3d_piont_list
        the dist is not considered at present.
    '''
    img_height, img_width = depth_img.shape
    u_array = np.array(range(img_width), dtype=np.float32)
    v_array = np.array(range(img_height), dtype=np.float32)
    one = np.ones((img_height, img_width, 1), dtype=np.float32)
    tmp1, tmp2 = np.meshgrid(u_array, v_array)
    
    """shape = (img_height, img_width, 1)
    content is like: 
    [
        [[0], [1]],
        [[0], [1]]
    ]
    """
    u_array = np.expand_dims(tmp1, axis=2)
    
    """shape = (img_height, img_width, 1)
    content is like: 
    [
        [[0], [0]],
        [[1], [1]]
    ]
    """
    v_array = np.expand_dims(tmp2, axis=2)

    camera_mat_i = camera_mat.I
    result_x = camera_mat_i[0,0] * u_array + camera_mat_i[0,1] * v_array + camera_mat_i[0,2] * one
    result_y = camera_mat_i[1,0] * u_array + camera_mat_i[1,1] * v_array + camera_mat_i[1,2] * one
    result_z = camera_mat_i[2,0] * u_array + camera_mat_i[2,1] * v_array + camera_mat_i[2,2] * one
    
    _3d_img = np.expand_dims(depth_img, axis=2) * np.concatenate([result_x, result_y, result_z], axis=2)
                
    return _3d_img

def get_pixel_index_array(img_width, img_height):
    """return a nparray like
    [
        [0,0,1],[1,0,1],
        [0,1,1],[1,1,1]
    ]
    """
    u_array = np.array(range(img_width), dtype=np.float32)
    v_array = np.array(range(img_height), dtype=np.float32)
    one = np.ones((img_height, img_width, 1), dtype=np.float32)
    tmp1, tmp2 = np.meshgrid(u_array, v_array)
    tmp1 = np.expand_dims(tmp1, axis=2)
    tmp2 = np.expand_dims(tmp2, axis=2)
    return np.concatenate([tmp1, tmp2, one], axis=2)
