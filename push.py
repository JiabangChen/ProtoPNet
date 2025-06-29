import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval() # ppnet_multi进入eval模式 不是train
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # 形成一个维度是2000的向量，每一个element都是无穷大，用这个来存储每一个P与最近的patch的距离
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])
    # 这个全0矩阵的形状是（2000，512，1，1），意思是用来存储对于每一个P而言distance最近的feature map patch
    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1) #初始化proto_rf_boxes和proto_bound_boxes，形状都为2000x6，
        # 每一列作用见上文绿字，rf_boxes是把与P最近的那个patch对应在原图上的rf的像素位置记录下来，bound_boxes是把P与和P最近的那个patch对应的
        # feature map的相似性分数上采样到原图后取出来的前5%区域的位置
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir) # 每做一次pushing时，把P对应的image patch存储到img_dir中，由于从第十个开始每过十个epoch就会
            # 做一次push，因此这是按照epoch_number存储的，如果epoch_number为None，之前存储的image patch就会被不断覆写
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size # 为了对每一张图都标上index，从而知道P是project到具体哪一张图
        # 找出与每个P最近的feature map patch是用train_push_loader来把train image输入到model中去的，每次按找batch输入，因此每个batch
        # 给image的index从push_iter * search_batch_size开始
        update_prototypes_on_batch(search_batch_input, # input
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y, # label
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)
    # 保存proto_rf_boxes和proto_bound_boxes，以numpy的格式
    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    # 相当于是把模型中的prototype vectors原地复制为新的P，即project后的P，而不能直接prototype_vectors = some_tensor，这样会损失掉
    # prototype_vectors原本的nn.Parameters的性质，即不再是一个可学习的模型参数
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))

# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input) # 这里还是用mean和std对输入做了normalization

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)
        # 当输入的是一个batch时，计算每一个sample经过backbone和add_on_layer后的feature map （B, 512, H, W）
        # 与这个map中的每一个pixel对每个P的距离 (B, 2000, H, W)
    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # 建立一个字典，每一个key是类别，值是一个空列表，用来存储输入batch中的每个sample在这个batch中的index（某个sample的index
        # 存到它的label（键）对应的列表（值）中）
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3] # P与feature map每个pixel的最大距离还是设置的是512

    for j in range(n_prototypes): # 对每个P最循环
        #if n_prototypes_per_class != None:
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
            # 找出第j个P对应的class类别
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue # 若是这个batch中的samples没有属于这个P的类的，那么就看下一个P
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]
            # class_to_img_index_dict[target_class] 指的是第j个P所属类在这个batch中的sample的index
            # proto_dist_[class_to_img_index_dict[target_class]]指的是这个batch的sample所计算出来的distance map（B，2000，H,W）
            # 的某几张distance map，根据index来挑选,比如index是1，3，5，那么输出shape就是（3，2000，H，W），共有6000张distance mao
            # proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]指的是在这几张distance map中，选取与第j个P的distance map
            # 比如只有三张，但是包含了feature map中的每一个pixel对P的距离
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:,j,:,:]

        batch_min_proto_dist_j = np.amin(proto_dist_j)# 对proto_dist_j整个矩阵找出最小值，即最小distance
        if batch_min_proto_dist_j < global_min_proto_dist[j]:#若这个最小distance确实比之前最小的distance还小
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            # 指找到最小值的索引然后将其位置转换成输入数据的形状格式，比如proto_dist_的形状是（3,16,16），最小值所在的索引是257，即第二张
            # 的第一个元素，那么最后输出最小值的位置就是[1，0，0]
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]
                # 因为知道是那三张图中的第二张有一个与P距离最小的patch是没有意义的，所以将batch_argmin_proto_dist_j的第一个元素转换成
                # 在这个batch中，P所属的那一类的哪一张图有与P距离最小的patch，相当于直接反馈出了这张图在batch中的index，这才有意义
            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w
            # 锁定了在这个batch中，与P最近的patch所属的图的index，以及这个patch在feature map中的位置。
            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]
            # 锁定了在这个batch中，根据与P最近的patch所属的图的index，以及这个patch在feature map中的位置，把这个patch具体数值给找出来
            # 形状是（512，）维的一个vector
            global_min_proto_dist[j] = batch_min_proto_dist_j # 更新第j个P与最近patch的距离
            global_min_fmap_patches[j] = batch_min_fmap_patch_j # 更新与第j个P最近的patch
            
            # get the receptive field boundary of the image patch
            # that generates the representation
            protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            # search_batch.size(2)是输入图像的行数，用来当作img_size，batch_argmin_proto_dist_j是与P最近的patch的位置（batch中
            # 第index张输入图中的第H，W位置的patch），这是为了获得与P最近的那个patch在input image上的rf位置，当然仍然保存了
            # 这个patch所属sample在batch中的index
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            # 根据patch所属sample在batch中的index找出对应原图（未normalize的），其大小为（3，H，W）
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0)) # 形状改为（H,W,3）
            original_img_size = original_img_j.shape[0]
            
            # crop out the receptive field，在原图中把rf切出来
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch # 真实index，即patch所属的图在
            # train_push_dataset中的index，而不是在当前batch中
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4] # patch对应在input image上的rf的位置
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item() # 这个image/这个P所属的类

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :] # 在batch内的distance map中选出此P的distance map，
            # 形状为（B，H，W）再进一步选出与此P最近的那个patch所属的样本的distance map，是一个二维的map
            if prototype_network_parallel.module.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.module.epsilon))
            elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            # 计算相似性分数map
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            # 相似性分数向上插值到原图大小
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]
            # 在原图中找出来P的activation map/similarity map上采样到原图大小后，前5%的区域
            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3] # activation map上采样到原图后的前5%区域
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)# 保存feature map大小的activation map，用numpy格式
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png，把包含P的那个patch的原图保存下来
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)# 因为本来这个原图就是totensor了，其值范围是在0，1之间，最后两个变量的作用是
                    # 强制将所有pixel按照0，1映射，即pixel value为0，就是0，为1，就是255，不加则会把这个原图的最小值映射为0，
                    # 这个原图的最大值映射为255，导致不同图像的值保存出来的颜色对比失真
                    # overlay (upsampled) self activation on original image and save the result
                    # 把activation map压在原图上保存下来
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    # Max-Min归一化
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    
                    # if different from the original (whole) image, save the prototype receptive field as png
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        # P投射过去的那个patch在原图上的rf区域保存下来，这可以当作是P的rf
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                      rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        # 根据patch的rf位置坐标来裁剪这个overlay图，并保存
                    # save the prototype image (highly activated region of the whole image)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    # 保存P的可视化图，就是activation map的前5%的区域，然后根据这个区域位置在原图上裁剪出来P的可视化patch
    if class_specific:
        del class_to_img_index_dict
