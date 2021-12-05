import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib

def resize_data(data, target_size=[128, 128, 128]):
    """
    Resize the data to the input size
    """
    [depth, height, width] = data.shape
    scale = [target_size[0] * 1.0 / depth, target_size[1] * 1.0 / height, target_size[2] * 1.0 / width]
    if len(data.shape) == 3:
        data = data.unsqueeze(0).unsqueeze(0)
    elif len(data.shape) == 4:
        data = data.unsqueeze(0)
    data = F.interpolate(data, scale_factor=scale)
    return data.squeeze()


def hilbert_3dto2d(data_3d, save=False):
    """
    data_3d: input 3d array or tensor with size 64*64*64
    """
    data_3d = data_3d.squeeze()
    if isinstance(data_3d, np.ndarray):
        data_2d = np.zeros([512, 512])
    elif isinstance(data_3d, torch.Tensor):
        data_2d = torch.zeros([512, 512])
    map = np.load('/root/data_2/fanchenchen/GAN/hilbert_map_64*64*64_to_512*512.npz')
    # points_3d, points_2d = map['points_3d'], map['points_2d']
    points_2d_index_x, points_2d_index_y = map['points_2d_index_x'], map['points_2d_index_y']
    points_3d_index_x, points_3d_index_y, points_3d_index_z = map['points_3d_index_x'], map['points_3d_index_y'], map[
        'points_3d_index_z']

    # Here are how to get those points
    # points_2d_index_x = tuple([points_2d[i][0] for i in range(points_2d.shape[0])])
    # points_2d_index_y = tuple([points_2d[i][1] for i in range(points_2d.shape[0])])
    #
    # points_3d_index_x = tuple([points_3d[i][0] for i in range(points_3d.shape[0])])
    # points_3d_index_y = tuple([points_3d[i][1] for i in range(points_3d.shape[0])])
    # points_3d_index_z = tuple([points_3d[i][2] for i in range(points_3d.shape[0])])

    # for index_put
    points_2d_index = (torch.LongTensor(list(points_2d_index_x)), torch.LongTensor(list(points_2d_index_y)))
    # points_3d_index = (torch.LongTensor(list(points_3d_index_x)), torch.LongTensor(list(points_3d_index_y)),
    #                    torch.LongTensor(list(points_3d_index_z)))

    data_1d = data_3d[points_3d_index_x, points_3d_index_y, points_3d_index_z].cpu()
    data_2d = data_2d.index_put(points_2d_index, data_1d)

    # Old slow method
    # for point_3d, point_2d in zip(points_3d, points_2d):
    #     data_2d[point_2d[0], point_2d[1]] = data_3d[point_3d[0], point_3d[1], point_3d[2]]
    if save:
        img_array = tensor2im(data_2d.unsqueeze(0), normalize=False)
        save_image(img_array, '/root/Downloads/test_hilbert_3dto2d.png')
    return data_2d.unsqueeze(0)


def hilbert_2dto3d(data_2d, save=False):
    """
    data_2d: input 2d array or tensor with size 512*512
    """
    if isinstance(data_2d, np.ndarray):
        data_3d = np.zeros([64, 64, 64])
    elif isinstance(data_2d, torch.Tensor):
        data_3d = torch.zeros([64, 64, 64])
    if len(data_2d.shape) > 2:
        data_2d = data_2d.squeeze()
    map = np.load('/root/data_2/fanchenchen/GAN/hilbert_map_64*64*64_to_512*512.npz')
    # points_3d, points_2d = map['points_3d'], map['points_2d']
    points_2d_index_x, points_2d_index_y = map['points_2d_index_x'], map['points_2d_index_y']
    points_3d_index_x, points_3d_index_y, points_3d_index_z = map['points_3d_index_x'], map['points_3d_index_y'], map[
        'points_3d_index_z']

    # Here are how to get those points
    # points_2d_index_x = tuple([points_2d[i][0] for i in range(points_2d.shape[0])])
    # points_2d_index_y = tuple([points_2d[i][1] for i in range(points_2d.shape[0])])
    #
    # points_3d_index_x = tuple([points_3d[i][0] for i in range(points_3d.shape[0])])
    # points_3d_index_y = tuple([points_3d[i][1] for i in range(points_3d.shape[0])])
    # points_3d_index_z = tuple([points_3d[i][2] for i in range(points_3d.shape[0])])

    # for index_put
    # points_2d_index = (torch.LongTensor(list(points_2d_index_x)), torch.LongTensor(list(points_2d_index_y)))
    points_3d_index = (torch.LongTensor(list(points_3d_index_x)), torch.LongTensor(list(points_3d_index_y)),
                       torch.LongTensor(list(points_3d_index_z)))

    data_1d = data_2d[points_2d_index_x, points_2d_index_y]
    data_3d = data_3d.index_put(points_3d_index, data_1d.cpu())

    # Old slow method
    # for point_3d, point_2d in zip(points_3d, points_2d):
    #     data_3d[point_3d[0], point_3d[1], point_3d[2]] = data_2d[point_2d[0], point_2d[1]]
    if save:
        image_nii = nib.Nifti1Image(data_3d.squeeze().numpy(), np.eye(4))
        nib.save(image_nii, '/root/Downloads/test_hilbert_2dto3d.nii.gz')
    data_3d = resize_data(data_3d, target_size=[128, 128, 128])
    return data_3d.unsqueeze(0)


def hilbert_3dto2d_cut(data_3d, stride_size=64, save=False):
    """
    data_3d: input 3d array or tensor with size 1*128*128*128
    """

    if isinstance(data_3d, np.ndarray):
        return_array = True
        data_3d = torch.from_numpy(data_3d)
    elif isinstance(data_3d, torch.Tensor):
        return_array = False

    if len(data_3d.shape) == 3:
        data_3d = data_3d.unsqueeze(0)
    elif len(data_3d.shape) == 5:
        data_3d = data_3d.squeeze(0)
    assert len(data_3d.shape) == 4, "Must input array/tensor with shape 1*128*128*128"
    patch_size = 64
    patches_data = data_3d.unfold(1, patch_size, stride_size
                                  ).unfold(2, patch_size, stride_size
                                           ).unfold(3, patch_size, stride_size).contiguous().cpu()
    patches_data = patches_data.view(-1, patch_size, patch_size, patch_size)

    data_2d = torch.zeros([patches_data.shape[0], 512, 512])

    map = np.load('/root/data_2/fanchenchen/GAN/hilbert_map_64*64*64_to_512*512.npz')
    # points_3d, points_2d = map['points_3d'], map['points_2d']
    points_2d_index_x, points_2d_index_y = map['points_2d_index_x'], map['points_2d_index_y']
    points_3d_index_x, points_3d_index_y, points_3d_index_z = map['points_3d_index_x'], map['points_3d_index_y'], map[
        'points_3d_index_z']

    # Here are how to get those points
    # points_2d_index_x = tuple([points_2d[i][0] for i in range(points_2d.shape[0])])
    # points_2d_index_y = tuple([points_2d[i][1] for i in range(points_2d.shape[0])])
    #
    # points_3d_index_x = tuple([points_3d[i][0] for i in range(points_3d.shape[0])])
    # points_3d_index_y = tuple([points_3d[i][1] for i in range(points_3d.shape[0])])
    # points_3d_index_z = tuple([points_3d[i][2] for i in range(points_3d.shape[0])])

    # for index_put
    points_2d_index = (torch.LongTensor(list(points_2d_index_x)), torch.LongTensor(list(points_2d_index_y)))
    # points_3d_index = (torch.LongTensor(list(points_3d_index_x)), torch.LongTensor(list(points_3d_index_y)),
    #                    torch.LongTensor(list(points_3d_index_z)))

    for i in range(patches_data.shape[0]):  # loop for 8 patchs
        data_3d_patch = patches_data[i]
        data_2d_patch = torch.zeros([512, 512])
        # data_1d = data_2d[points_2d_index_x, points_2d_index_y]
        data_1d = data_3d_patch[points_3d_index_x, points_3d_index_y, points_3d_index_z]
        data_2d_patch = data_2d_patch.index_put(points_2d_index, data_1d)
        data_2d[i] = data_2d_patch
    # Old slow method
    # for point_3d, point_2d in zip(points_3d, points_2d):
    #     data_2d[:, point_2d[0], point_2d[1]] = patches_data[:, point_3d[0], point_3d[1], point_3d[2]]

    if save:
        for i in range(data_2d.shape[0]):
            img_array = tensor2im(data_2d[i].unsqueeze(0), normalize=False)
            save_image(img_array, '/root/Downloads/test_hilbert_3dto2d_cut_{}.png'.format(i))
    if return_array:
        return data_2d.numpy().unsqueeze(0)
    else:
        return data_2d.unsqueeze(0)


def hilbert_2dto3d_cut(data_2d, stride_size=64, save=False):
    """
    data_2d: input 2d array or tensor with size N*512*512
    return: data_3d 1*128*128*128
    """
    if isinstance(data_2d, np.ndarray):
        return_array = True
    elif isinstance(data_2d, torch.Tensor):
        return_array = False
    # print(data_2d.shape)
    data_2d = data_2d.squeeze()
    patches_data = torch.zeros([data_2d.shape[0], 64, 64, 64])

    map = np.load('/root/data_2/fanchenchen/GAN/hilbert_map_64*64*64_to_512*512.npz')
    # points_3d, points_2d = map['points_3d'], map['points_2d']
    points_2d_index_x, points_2d_index_y = map['points_2d_index_x'], map['points_2d_index_y']
    points_3d_index_x, points_3d_index_y, points_3d_index_z = map['points_3d_index_x'], map['points_3d_index_y'], map[
        'points_3d_index_z']

    # Here are how to get those points
    # points_2d_index_x = tuple([points_2d[i][0] for i in range(points_2d.shape[0])])
    # points_2d_index_y = tuple([points_2d[i][1] for i in range(points_2d.shape[0])])
    #
    # points_3d_index_x = tuple([points_3d[i][0] for i in range(points_3d.shape[0])])
    # points_3d_index_y = tuple([points_3d[i][1] for i in range(points_3d.shape[0])])
    # points_3d_index_z = tuple([points_3d[i][2] for i in range(points_3d.shape[0])])

    # for index_put
    # points_2d_index = (torch.LongTensor(list(points_2d_index_x)), torch.LongTensor(list(points_2d_index_y)))
    points_3d_index = (torch.LongTensor(list(points_3d_index_x)), torch.LongTensor(list(points_3d_index_y)),
                       torch.LongTensor(list(points_3d_index_z)))

    for i in range(patches_data.shape[0]):  # loop for 8 patchs
        data_2d_patch = data_2d[i].cpu()
        data_3d_patch = torch.zeros([64, 64, 64])
        # data_1d = data_2d[points_2d_index_x, points_2d_index_y]
        data_1d = data_2d_patch[points_2d_index_x, points_2d_index_y]
        data_3d_patch = data_3d_patch.index_put(points_3d_index, data_1d)
        patches_data[i] = data_3d_patch
    # Old slow method
    # for point_3d, point_2d in zip(points_3d, points_2d):
    #     patches_data[:, point_3d[0], point_3d[1], point_3d[2]] = data_2d[:, point_2d[0], point_2d[1]]

    patches_data = patches_data.view(1, 2, 2, 2, 64, 64, 64)
    cat_x = torch.cat((patches_data[:, 0, :, :, :, :, :], patches_data[:, 1, :, :, :, :, :]), dim=3)
    cat_xy = torch.cat((cat_x[:, 0, :, :, :, :], cat_x[:, 1, :, :, :, :]), dim=3)
    cat_xyz = torch.cat((cat_xy[:, 0, :, :, :, ], cat_xy[:, 1, :, :, :]), dim=3).contiguous()
    if save:
        for i in range(8):
            image_nii = nib.Nifti1Image(patches_data.view(8, 64, 64, 64)[i].squeeze().numpy(), np.eye(4))
            nib.save(image_nii, '/root/Downloads/test_hilbert_2dto3d_cut_{}.nii.gz'.format(i))
        image_nii = nib.Nifti1Image(cat_xyz.squeeze().numpy(), np.eye(4))
        nib.save(image_nii, '/root/Downloads/test_hilbert_2dto3d_cut_whole.nii.gz')
    if return_array:
        return cat_xyz.numpy()
    else:
        return cat_xyz


def linear_3dto2d(data_3d, save=False):
    """
    data_3d: input 3d array or tensor with size 64*64*64
    """
    data_3d = data_3d.squeeze()
    data_2d = data_3d.view(512, 512)
    if save:
        img_array = tensor2im(data_2d.unsqueeze(0), normalize=False)
        save_image(img_array, '/root/Downloads/test_linear_3dto2d.png')
    return data_2d.unsqueeze(0)


def linear_2dto3d(data_2d, save=False):
    """
    data_2d: input 2d array or tensor with size 512*512
    """
    data_3d = data_2d.view(64, 64, 64)
    if save:
        image_nii = nib.Nifti1Image(data_3d.squeeze().numpy(), np.eye(4))
        nib.save(image_nii, '/root/Downloads/test_linear_2dto3d.nii.gz')
    data_3d = resize_data(data_3d, target_size=[128, 128, 128])
    return data_3d.unsqueeze(0)


def linear_3dto2d_cut(data_3d, stride_size=64, save=False):
    """
    data_3d: input 3d array or tensor with size 1*128*128*128
    """

    if isinstance(data_3d, np.ndarray):
        return_array = True
        data_3d = torch.from_numpy(data_3d)
    elif isinstance(data_3d, torch.Tensor):
        return_array = False

    if len(data_3d.shape) == 3:
        data_3d = data_3d.unsqueeze(0)
    elif len(data_3d.shape) == 5:
        data_3d = data_3d.squeeze(0)
    assert len(data_3d.shape) == 4, "Must input array/tensor with shape 1*128*128*128"
    patch_size = 64
    patches_data = data_3d.unfold(1, patch_size, stride_size
                                  ).unfold(2, patch_size, stride_size
                                           ).unfold(3, patch_size, stride_size).contiguous()
    data_2d = patches_data.view(-1, 512, 512)
    if save:
        for i in range(data_2d.shape[0]):
            img_array = tensor2im(data_2d[i].unsqueeze(0), normalize=False)
            save_image(img_array, '/root/Downloads/test_linear_3dto2d_cut_{}.png'.format(i))

    if return_array:
        return data_2d.numpy().unsqueeze(0)
    else:
        return data_2d.unsqueeze(0)


def linear_2dto3d_cut(data_2d, stride_size=64, save=False):
    """
    data_2d: input 2d array or tensor with size N*512*512
    """
    if isinstance(data_2d, np.ndarray):
        return_array = True
    elif isinstance(data_2d, torch.Tensor):
        return_array = False
    data_2d = data_2d.squeeze()
    patches_data = data_2d.view(1, 2, 2, 2, 64, 64, 64)
    cat_x = torch.cat((patches_data[:, 0, :, :, :, :, :], patches_data[:, 1, :, :, :, :, :]), dim=3)
    cat_xy = torch.cat((cat_x[:, 0, :, :, :, :], cat_x[:, 1, :, :, :, :]), dim=3)
    cat_xyz = torch.cat((cat_xy[:, 0, :, :, :, ], cat_xy[:, 1, :, :, :]), dim=3).contiguous()
    if save:
        for i in range(8):
            image_nii = nib.Nifti1Image(patches_data.view(8, 64, 64, 64)[i].squeeze().numpy(), np.eye(4))
            nib.save(image_nii, '/root/Downloads/test_linear_2dto3d_cut_{}.nii.gz'.format(i))
        image_nii = nib.Nifti1Image(cat_xyz.squeeze().numpy(), np.eye(4))
        nib.save(image_nii, '/root/Downloads/test_linear_2dto3d_cut_whole.nii.gz')
    if return_array:
        return cat_xyz.numpy()
    else:
        return cat_xyz
