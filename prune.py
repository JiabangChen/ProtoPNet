import os
import shutil
from collections import Counter
import numpy as np
import torch

from helpers import makedir
import find_nearest

def prune_prototypes(dataloader,
                     prototype_network_parallel,
                     k,
                     prune_threshold,
                     preprocess_input_function,
                     original_model_dir,
                     epoch_number,
                     #model_name=None,
                     log=print,
                     copy_prototype_imgs=True):
    ### run global analysis
    nearest_train_patch_class_ids = \
        find_nearest.find_k_nearest_patches_to_prototypes(dataloader=dataloader,
                                                          prototype_network_parallel=prototype_network_parallel,
                                                          k=k,
                                                          preprocess_input_function=preprocess_input_function,
                                                          full_save=False,
                                                          log=log)
    # 提取与每个P距离最小的6个patch的所属label
    ### find prototypes to prune
    original_num_prototypes = prototype_network_parallel.module.num_prototypes # P的数量
    
    prototypes_to_prune = []
    for j in range(prototype_network_parallel.module.num_prototypes):
        class_j = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item() # P的类别
        nearest_train_patch_class_counts_j = Counter(nearest_train_patch_class_ids[j])
        # if no such element is in Counter, it will return 0
        if nearest_train_patch_class_counts_j[class_j] < prune_threshold:
            prototypes_to_prune.append(j)
        # 如果与P距离最小的6个patch所属类别有>3个不是P它所属的类别，那么就说这个P该被prune了 这里在做prune
    log('k = {}, prune_threshold = {}'.format(k, prune_threshold))
    log('{} prototypes will be pruned'.format(len(prototypes_to_prune)))

    ### bookkeeping of prototypes to be pruned 记录要被prune掉的P的index和类别
    class_of_prototypes_to_prune = \
        torch.argmax(
            prototype_network_parallel.module.prototype_class_identity[prototypes_to_prune],
            dim=1).numpy().reshape(-1, 1)
        # prototype_class_identity的形状是(2000,200)，prototype_class_identity[prototypes_to_prune]的形状是(N,200),假设
    # 有N个P要被修剪掉，这个代码的意思是找出要被修剪掉的P的类别，然后reshape为（N，1）
    prototypes_to_prune_np = np.array(prototypes_to_prune).reshape(-1, 1) # prototypes_to_prune也reshape为(N,1)
    prune_info = np.hstack((prototypes_to_prune_np, class_of_prototypes_to_prune))
    # prune_info的形状是(N,2),第一列代表要被裁剪掉的P的index，第二列代表P的类别，并把它用numpy的形式存放
    makedir(os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number,
                                          k,
                                          prune_threshold)))
    np.save(os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number,
                                          k,
                                          prune_threshold), 'prune_info.npy'),
            prune_info)

    ### prune prototypes
    prototype_network_parallel.module.prune_prototypes(prototypes_to_prune) # 调用模型中的prune_prototypes这个方法来修剪P
    # 这是只保存参数字典所做不到的
    #torch.save(obj=prototype_network_parallel.module,
    #           f=os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number,
    #                                              k,
    #                                              prune_threshold),
    #                          model_name + '-pruned.pth'))
    if copy_prototype_imgs:
        original_img_dir = os.path.join(original_model_dir, 'img', 'epoch-%d' % epoch_number)
        dst_img_dir = os.path.join(original_model_dir,
                                   'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch_number,
                                                                               k,
                                                                               prune_threshold),
                                   'img', 'epoch-%d' % epoch_number)
        makedir(dst_img_dir)
        prototypes_to_keep = list(set(range(original_num_prototypes)) - set(prototypes_to_prune))
        # 把不需要prune掉的P的索引形成一个列表
        for idx in range(len(prototypes_to_keep)):
            shutil.copyfile(src=os.path.join(original_img_dir, 'prototype-img%d.png' % prototypes_to_keep[idx]),
                            dst=os.path.join(dst_img_dir, 'prototype-img%d.png' % idx))
            
            shutil.copyfile(src=os.path.join(original_img_dir, 'prototype-img-original%d.png' % prototypes_to_keep[idx]),
                            dst=os.path.join(dst_img_dir, 'prototype-img-original%d.png' % idx))
            
            shutil.copyfile(src=os.path.join(original_img_dir, 'prototype-img-original_with_self_act%d.png' % prototypes_to_keep[idx]),
                            dst=os.path.join(dst_img_dir, 'prototype-img-original_with_self_act%d.png' % idx))
            
            shutil.copyfile(src=os.path.join(original_img_dir, 'prototype-self-act%d.npy' % prototypes_to_keep[idx]),
                            dst=os.path.join(dst_img_dir, 'prototype-self-act%d.npy' % idx))
            # 把不需要prune掉的P的可视，包含P的那个patch的原图，P对它对应的patch的feature map产生的activation map（numpy格式），和这个
            # activation map上采样然后形成的overlay图保存

            bb = np.load(os.path.join(original_img_dir, 'bb%d.npy' % epoch_number))
            bb = bb[prototypes_to_keep]
            np.save(os.path.join(dst_img_dir, 'bb%d.npy' % epoch_number),
                    bb)

            bb_rf = np.load(os.path.join(original_img_dir, 'bb-receptive_field%d.npy' % epoch_number))
            bb_rf = bb_rf[prototypes_to_keep]
            np.save(os.path.join(dst_img_dir, 'bb-receptive_field%d.npy' % epoch_number),
                    bb_rf)
            # 把proto_rf_boxes和proto_bound_boxes中不需要被prune掉的P的信息，保存下来
    return prune_info
