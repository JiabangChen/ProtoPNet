##### MODEL AND DATA LOADING
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import re

import os
import copy

from helpers import makedir, find_high_activation_crop
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse
###############################Jiabang's alert，如果用prune后的model来做，看下prune_alert:后的注释##################################

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-imgdir', nargs=1, type=str)
parser.add_argument('-img', nargs=1, type=str)
parser.add_argument('-imgclass', nargs=1, type=int, default=-1)
args = parser.parse_args() # Jiabang's alert 跑之前要输入这些参数

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
for num in range(0,200):
    # specify the test image to be analyzed
    # 按照作者原注释来输入，这里只需要输入一张测试图来做测试就好，但需要新建一个local_analysis/Painted_Bunting_Class15_0081文件夹来存放对应的图像
    # test_image_label记得减去1    Jiabang's alert
    test_image_dir = args.imgdir[0] #'./local_analysis/Painted_Bunting_Class15_0081/'
    test_image_name = args.img[0] #'Painted_Bunting_0081_15230.jpg'
    test_image_label = args.imgclass[0] #15
    local_analysis_path = './local_analysis/'
    test_image_dir = os.path.join(local_analysis_path, os.listdir(local_analysis_path)[num])
    test_image_name = os.listdir(test_image_dir)[0]
    test_image_label = num
    print(test_image_dir)
    print(test_image_name)
    print(test_image_label)
    test_image_path = os.path.join(test_image_dir, test_image_name) # test image来源路径

    # load the model
    check_test_accu = False
    # 同样 找一个accuracy合适的model，最好是刚做完push的，因为如果是joint之后的，P和feature map的值又发生了变化，这时与test image最近的P的
    # 可视化图表征的信息和P实际的信息可能有差距，因为这时P已经不是刚push完的P了，而是做了优化了 Jiabang's alert
    load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/'
    load_model_name = args.model[0] #'10_18push0.7822.pth'
    # prune_alert: 如果用prune后的model来做,那么模型dir和name要变，得用prune后的模型

    #if load_model_dir[-1] == '/':
    #    model_base_architecture = load_model_dir.split('/')[-3]
    #    experiment_run = load_model_dir.split('/')[-2]
    #else:
    #    model_base_architecture = load_model_dir.split('/')[-2]
    #    experiment_run = load_model_dir.split('/')[-1]

    model_base_architecture = load_model_dir.split('/')[2]
    # 若输入是'./saved_models/vgg19/003/' 这里是vgg19，具体按照输入来写是第几个元素，Jiabang's alert
    experiment_run = '/'.join(load_model_dir.split('/')[3:4]) # 输出003，但同样 这里也需要按照具体输入来看是哪个元素 Jiabang's alert

    save_analysis_path = os.path.join(test_image_dir, model_base_architecture,
                                      experiment_run, load_model_name) # 存放测试图的the nearest prototypes路径，并带上log
    makedir(save_analysis_path)

    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

    load_model_path = os.path.join(load_model_dir, load_model_name) # model加载路径
    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    # 找出这是第几个epoch的model，注意不是一个epoch中单独跑FC的第几个轮次
    start_epoch_number = int(epoch_number_str)

    log('load model from ' + load_model_path)
    log('model base architecture: ' + model_base_architecture)
    log('experiment run: ' + experiment_run)

    '''
    ppnet = torch.load(load_model_path, weights_only=False) # 这里就告诉了要如何加载一个训练好了的ppnet
    # Jiabang's change, plus weights_only=False can load the model successfully
    ppnet = ppnet.cuda() 
    '''
    # jiabang's alert 如果要放在cpu上跑，上面两行代码要改为下面两行代码
    ppnet = torch.load(load_model_path, weights_only=False, map_location=torch.device('cpu'))
    ppnet.to('cpu')
    ppnet_multi = torch.nn.DataParallel(ppnet)

    img_size = ppnet_multi.module.img_size
    prototype_shape = ppnet.prototype_shape # 这些信息就要save model才可以调取，如果只保存模型的参数字典，就无法调取
    # prune: 如果用prune后的model来做，那么prototype_shape会变，但max_dist不变
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3] # 还是算出最大的距离，比如512

    class_specific = True

    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    # load the test data and check test accuracy
    from settings import test_dir
    if check_test_accu:
        test_batch_size = 100

        test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=True)
        # Jiabang's change,删去了num_workers=4, pin_memory=False
        log('test set size: {0}'.format(len(test_loader.dataset)))

        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=print)

    ##### SANITY CHECK
    # confirm prototype class identity
    load_img_dir = os.path.join(load_model_dir, 'img')
    #prune_alert: 如果用prune后的模型来做local_analysis的话，存放P的文件夹要变，load_img_dir要变为pruned_prototypes_epoch10_k6_pt3
    # 找到专门存放P的文件夹
    prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
    # prune: 如果用prune后的model来做，这个prototype_info就会变
    # 把在某个epoch做push时的proto_bound_boxes拿出来，是一个2000x6的表格
    prototype_img_identity = prototype_info[:, -1]# 意思是proto_bound_boxes的最后一列，即每个P所属的类别

    log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
    log('Their class identities are: ' + str(prototype_img_identity))

    # confirm prototype connects most strongly to its own class
    prototype_max_connection = torch.argmax(ppnet.last_layer.weight, dim=0)
    # prune: 如果用prune后的model来做，最后的FC层权重也会变
    # 最后一层weight的形状是(200，2000),那么取最大的意思是看这个P所计算出来的相似性分数主要对哪一类有正向作用，它理应对自己所属的类有最大的正向作用
    prototype_max_connection = prototype_max_connection.cpu().numpy()
    if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
        log('All prototypes connect most strongly to their respective classes.')
    else:
        log('WARNING: Not all prototypes connect most strongly to their respective classes.')
    # 如果所有的P所计算出来的相似值都是对自己的那个类有最大的正向作用，那这才是我们想要的，否则就要报警
    ##### HELPER FUNCTIONS FOR PLOTTING
    def save_preprocessed_img(fname, preprocessed_imgs, index=0):
        img_copy = copy.deepcopy(preprocessed_imgs[index:index+1]) # 把用于找出最近P的第一张测试图找出来，但这时不去掉batch维度，因此
        # img_copy的形状仍然是（1，3，H,W）
        undo_preprocessed_img = undo_preprocess_input_function(img_copy) # 把mean和std做的normalization从图中去掉
        print('image index {0} in batch'.format(index))
        undo_preprocessed_img = undo_preprocessed_img[0] # undo_preprocessed_img形状是（3,H,W）
        undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
        undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0]) # 把形状改为（H,W,3）

        plt.imsave(fname, undo_preprocessed_img)
        return undo_preprocessed_img

    def save_prototype(fname, epoch, index):
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
        # 根据与feature map相似度分数最高的那个P的index，把这个P的可视化图，即P投射过去的那个patch所属的feature map的activation map前5%的区域找出来
        #plt.axis('off')
        plt.imsave(fname, p_img)

    def save_prototype_self_activation(fname, epoch, index):
        p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                        'prototype-img-original_with_self_act'+str(index)+'.png'))
        #plt.axis('off')
        plt.imsave(fname, p_img)

    def save_prototype_original_img_with_bbox(fname, epoch, index,
                                              bbox_height_start, bbox_height_end,
                                              bbox_width_start, bbox_width_end, color=(0, 255, 255)):
        p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
        cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                      color, thickness=2)
        p_img_rgb = p_img_bgr[...,::-1]
        p_img_rgb = np.float32(p_img_rgb) / 255
        #plt.imshow(p_img_rgb)
        #plt.axis('off')
        plt.imsave(fname, p_img_rgb)

    def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                         bbox_width_start, bbox_width_end, color=(0, 255, 255)): # 框是黄色的
        img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
        cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                      color, thickness=2)
        img_rgb_uint8 = img_bgr_uint8[...,::-1]
        img_rgb_float = np.float32(img_rgb_uint8) / 255
        #plt.imshow(img_rgb_float)
        #plt.axis('off')
        plt.imsave(fname, img_rgb_float)

    # load the test image and forward it through the network
    preprocess = transforms.Compose([
       transforms.Resize((img_size,img_size)),
       transforms.ToTensor(),
       normalize
    ])

    img_pil = Image.open(test_image_path)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))

    # images_test = img_variable.cuda() # 其实感觉是可以输入多张测试图的，但这里作者举例只用了一张图，但它的形状应该是（1，3，H,W）
    # jiabang's alert 如果要放在cpu上跑，上行代码要改为下行代码
    images_test = img_variable.cpu()
    labels_test = torch.tensor([test_image_label])

    logits, min_distances = ppnet_multi(images_test)# 这里的min_distance是所有P与这个输入的feature map的最小距离，形状是一个（1，2000）的矩阵
    conv_output, distances = ppnet.push_forward(images_test)
    # 一个是这张图输入后由backbone和add_on_layer产生的feature map，形状是（1，512，H，W），一个是这个feature map逐元素与各个P计算距离，形状是
    # （1，2000，H,W）
    # prune: 如果用prune后的model来做，那么min_distances和distances的形状都会变,同理prototype_activations和prototype_activation_patterns的形状也会变
    prototype_activations = ppnet.distance_2_similarity(min_distances)
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)
    # 把feature map与各个P的距离以及最短距离转为相似性分数
    if ppnet.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist
        prototype_activation_patterns = prototype_activation_patterns + max_dist

    tables = []
    for i in range(logits.size(0)): # logits.size(0)就是看有多少个图像输入，这里应只有一张图，logits的size应该是（1，200）
        tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
        # 找出logits中最大的索引，即模型所判为的类别，和实际类别
        log(str(i) + ' ' + str(tables[-1]))

    idx = 0
    predicted_cls = tables[idx][0]
    correct_cls = tables[idx][1]
    log('Predicted: ' + str(predicted_cls))
    log('Actual: ' + str(correct_cls))
    original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                         images_test, idx) # 目的是把用来找最近的P的测试原图保存下来

    ##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
    makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

    log('Most activated 10 prototypes of this image:')
    array_act, sorted_indices_act = torch.sort(prototype_activations[idx])# 之所以要[idx]是为了把batch维度消除掉
    # 当我算出这个输入图的feature map后，又算出了其对每个P的最小距离，并转化成了相似性分数，然后对相似性分数进行排序，第一个输出参数是排序后的结果
    # 第二个输出参数指的是原始张量中这些值的位置索引，即可得知相似性分数最高的那个P的index。这个排序应该是从小到大，最大的在最后
    # prune: 如果用prune后的model来做 那么上述两个输出都会变
    for i in range(1,11):
        log('top {0} activated prototype for this image:'.format(i))
        save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                    'top-%d_activated_prototype.png' % i),
                       start_epoch_number, sorted_indices_act[-i].item())# sorted_indices_act[-i].item()用来找出与测试图的feature
        # map相似度分数最高的十个P的index，然后把这十个P的可视图保存下来
        save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                                 'top-%d_activated_prototype_in_original_pimg.png' % i),
                                              epoch=start_epoch_number,
                                              index=sorted_indices_act[-i].item(),
                                              bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                              bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                              bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                              bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                              color=(0, 255, 255))
        # 这是把P的可视图用bounding box的形式，在P投射过去的那个patch所属的原图上标记出来，然后保存
        save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                    'top-%d_activated_prototype_self_act.png' % i),
                                       start_epoch_number, sorted_indices_act[-i].item())
        # 这个是在把与这个feature map距离最近/相似度分数最高的十个P在P对应的patch所属的原图上的activation map，然后overlay到原图上，然后保存下来
        # 以上保存的是与feature map相似性分数最高的十个P的信息，是P的信息（P的可视图，可视图的来源，overlay）
        log('prototype index: {0}'.format(sorted_indices_act[-i].item())) # 拥有最高相似性分数的P的index
        log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
        # 拥有最高相似性分数的P他们所属的类别
        if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
            # 如果拥有最高相似性分数的P所贡献度最高的那个类不等于其自身所属的类
            log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
        log('activation value (similarity score): {0}'.format(array_act[-i])) # 最高的十个相似性分数 是XAI的一部分
        log('last layer connection with predicted class: {0}'.format(ppnet.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
        # 这些对feature map具有高相似性的P在FC层对于分类的权重，这个类是预测类别，而非实际类别，也是XAI的一部分
        # prune: 如果用prune后的model来做,上述保存P的信息就需要从prune时转存的P的文件夹中去取


        activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
        # 这是找出输入sample的feature map与相似度最高的那十个P的activation map/similarity score map，形状为（H,W）
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                  interpolation=cv2.INTER_CUBIC) # 上采样到原图大小

        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                      high_act_patch_indices[2]:high_act_patch_indices[3], :]
        # 这是知道了十个P与此feature map的min_distance最小，然后来看下这些P在这个feature map上的activation map，然后上采样到原图大小
        # 然后再找出前5%的区域，作为high_act_patch
        log('most highly activated patch of the chosen image by this prototype:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                   high_act_patch) # 保存activation map中的前5%区域，注意这里的activation map是P对于输入feature map计算的，
        # 而非是P对于其投射过去的patch所属的feature map计算的
        log('most highly activated patch by this prototype shown in the original image:')
        imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                         img_rgb=original_img,
                         bbox_height_start=high_act_patch_indices[0],
                         bbox_height_end=high_act_patch_indices[1],
                         bbox_width_start=high_act_patch_indices[2],
                         bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        # 当我找出了P对于此输入feature map的activation map的前5%区域后，我用bounding box在输入测试原图上把该区域画出来，然后保存
        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern) # Min-Max归一化
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('prototype activation map of the chosen image:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'prototype_activation_map_by_top-%d_prototype.png' % i),
                   overlayed_img)
        # 这是在把与此测试图距离最近的P对于此测试图的activation map上采样到原图大小后overlay到测试原图上
        # 上述保存的是P作用在测试图上的信息（activation最高的区域，bounding box圈出来源，和overlay）
        # 上述的P并不一定就属于输入测试图的预测类，他们只是与输入测试图的similarity score最高的那十个
        log('--------------------------------------------------------------')
    '''
    ##### PROTOTYPES FROM TOP-k CLASSES
    k = 50
    log('Prototypes from top-%d classes:' % k)
    topk_logits, topk_classes = torch.topk(logits[idx], k=k) # 找出logits中分数最高的50个logits，和他们对应的类别
    for i,c in enumerate(topk_classes.detach().cpu().numpy()):
        makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))
    
        log('top %d predicted class: %d' % (i+1, c))
        log('logit of the class: %f' % topk_logits[i])
        class_prototype_indices = np.nonzero(ppnet.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
        # 返回一个ndarray，里面是此c类的P的index
        class_prototype_activations = prototype_activations[idx][class_prototype_indices]
        # 先算出我所输入的feature map对2000个P的最短距离，然后转成activation / similarity score，然后取出这中间属于c类的那几个P的activation值
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)
        # 相似性分数从小到大排序，并输出排序后的这十个属于c类的P的相似性分数在class_prototype_activations中的位置索引
    
        prototype_cnt = 1
        for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()): # reverse是为了按照相似性分数从大往小取
            prototype_index = class_prototype_indices[j] # 根据P在class_prototype_activations上的位置索引来找出P在所有P列表中的index
            save_prototype(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                        'top-%d_activated_prototype.png' % prototype_cnt),
                           start_epoch_number, prototype_index) # 把属于c类的P的可视图存一下
            save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                                     'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                                  epoch=start_epoch_number,
                                                  index=prototype_index,
                                                  bbox_height_start=prototype_info[prototype_index][1],
                                                  bbox_height_end=prototype_info[prototype_index][2],
                                                  bbox_width_start=prototype_info[prototype_index][3],
                                                  bbox_width_end=prototype_info[prototype_index][4],
                                                  color=(0, 255, 255))# 把属于c类的P的可视图，即那个patch在这个patch所属原图上用框框出来
            save_prototype_self_activation(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                        'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                           start_epoch_number, prototype_index)
            # 把属于c类的P在其对应的patch的原图上的activation map overlay图保存下来
            log('prototype index: {0}'.format(prototype_index)) # P在整体P列表中的index
            log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index])) # P的类别
            if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
                # 如果这个P贡献最多的那个类不是这个P自身所属的类
                log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
            log('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
            # 此P与输入feature map的相似性分数 XAI的组成部分
            log('last layer connection: {0}'.format(ppnet.last_layer.weight[c][prototype_index]))# 这个P对于其所属的类在FC层上贡献值
            # XAI的组成部分
            # 以上保存的是输入feature map最高的五十个预测logits对应的类的P的信息，是P的信息（P的可视图，可视图的来源，overlay）
            activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
            # 从输入的测试图的feature map对各个P的activation map中找出c类的P对应的那个activation map
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                      interpolation=cv2.INTER_CUBIC) # 上采样
            
            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                          high_act_patch_indices[2]:high_act_patch_indices[3], :]
            log('most highly activated patch of the chosen image by this prototype:')
            # 从输入feature map的最高五十个类中找出其中一个类的其中一个P，然后对输入feature map求activation map，上采样到原图大小后找出前5%
            # 的区域，然后对应在原图上把这个区域割出来
            #plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                       high_act_patch)
            # 保存activation map中的前5%区域，注意这里的activation map是P对于输入feature map计算的，
            # 而非是P对于其投射过去的patch所属的feature map计算的
            log('most highly activated patch by this prototype shown in the original image:')
            imsave_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                             img_rgb=original_img,
                             bbox_height_start=high_act_patch_indices[0],
                             bbox_height_end=high_act_patch_indices[1],
                             bbox_width_start=high_act_patch_indices[2],
                             bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
            # 当我找出了P对于此输入feature map的activation map的前5%区域后，我用bounding box在输入测试原图上把该区域画出来，然后保存
            # show the image overlayed with prototype activation map
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap
            log('prototype activation map of the chosen image:')
            #plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                       overlayed_img)
            # 这是在把此测试图分类最高的五十个类的其中一个类的其中一个P对于此测试图的activation map上采样到原图大小后overlay到测试原图上
            # 上述保存的是P作用在测试图上的信息（activation最高的区域，bounding box圈出来源，和overlay）
            # 上述的P属于输入测试图的预测类（前五十个）
            log('--------------------------------------------------------------')
            prototype_cnt += 1
        log('***************************************************************')
    '''
    # temporary not use this function Jiabang's change
    if predicted_cls == correct_cls:
        log('Prediction is correct.')
    else:
        log('Prediction is wrong.')

    logclose()

