import os
import shutil
###################################跑之前可以看下Jiabang's alert##################################3
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse # argparse 用于解析命令行参数
import re

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
# 创建一个命令行参数解析器对象parser
parser = argparse.ArgumentParser()
# 这里用来告诉应该如何解析：比如这里的命令行参数是-gpuid，这个命令行接受一个字符串输入（type=str)
# 且默认值是‘0’，且只接受一个字符串，因为nargs=1，而且这个会让args.gpuid变成一个列表，所以后需要加args.gpuid[0]才可以读取
# 这个nargs，type，default就类似于命令行参数的输入类型（输入数量，type，默认值）
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
# 我如果输入 python3 main.py -gpuid=0,1,2,3，那么相当于我施加了一个命令，参数为-gpuid，输入值是一个字符串是0，1，2，3，args.gpuid会返回['0,1,2,3']
# 我如果输入 python3 main.py，那么相当于我没有施加命令，args.gpuid是默认值['0']
args = parser.parse_args() # 从命令行中实际读取用户输入，并将其解析成一个对象 args。args的属性名就是命令的参数名，这里是gpuid，
# 对应的值就是用户在命令行中输入的值
# 至于这个命令去哪里下，因为我用的是pycharm就非常方便，直接右键main.py选择more run/debug，选择modify run configuration,然后在
# script parameters里输入-gpuid=0,1,2,3就好了
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0] # 指定pytorch用哪几块GPU，这里是0，1，2，3共四块GPU
print(os.environ['CUDA_VISIBLE_DEVICES']) # 在跑之前要查一下有几块GPU，一般我只有1块 Jiabang's alert

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0) # 把vgg19变成vgg
# re是正则表达式模块，

model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir) # 建立一个文件夹来save一些与model相关的文档
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir) # 把当前正在运行的文件保存的.py文件保存过去
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir) # 把settings.py文件保存过去
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)# 把vgg_features.py保存过去
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir) # 把model.py保存过去
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir) # 把train_and_test.py保存过去

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))# 在model_dir中创建一个log文档，写log
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir) # 在model文件夹中建立一个文件夹用于存放prototype
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb' # prefix：前缀

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# all datasets
# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True) # Jiabang's change,删去了num_workers=4, pin_memory=False
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False)# Jiabang's change,删去了num_workers=4, pin_memory=False
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False)# Jiabang's change,删去了num_workers=4, pin_memory=False

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet) # 让模型 ppnet 在多个 GPU 上并行执行，以加快训练或推理速度。主 GPU 保留“真实模型”（参数权重）
# 所有参数的更新都发生在主 GPU（默认是 cuda:0）上；其他GPU上也会复制模型，但只是临时副本，可以接受一个比较大的batch，然后把数据分发到每个GPU
# 上，在每张显卡上进行计算（前向+反向梯度计算，并把梯度发到主GPU上，combine之后做参数更新），但是参数更新会实时反馈到主GPU上，在后一轮
# forward时，比如后一轮batch时，其他GPU会从主GPU上再次复制model（这时的weight已经更新过了一次）
# 由于用DataParallel把ppnet包裹起来，因此直接用ppnet_multi（x)跑模型是可以的，但是如果想要接触这个模型中的layer和参数，就要
# ppnet_multi.module.xxx，而不是ppnet_multi.xxx
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
 # bias are now also being regularized，bias也受到weight——decay调节
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs) # 用Adam做optimizer，这个optimizer对应与training的第一个stage
# 即训练backbone+add_on_layers+Prototype层
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)
# scheduler是stepLR，每过5个epoch，lr*0.1
from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
# warm-up epoch时的optimizer
from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
# 对FC层训练的optimizer
# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs

# train the model
log('start training')
import copy
for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs: # 训练前先做warm-up training
        tnt.warm_only(model=ppnet_multi, log=log) # 调整参数训练与否
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step() # 因为这里才是第一次调用scheduler，所以这才是scheduler计数达到五次lr/10的起点，前面warm-up跑的
        # epoch不会对joint_optimizer的lr是否要/10产生影响
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.70, log=log)
    # 这里的model不是包在DataParallel中的model ppnet_multi，这里是把模型保存到model_dir中，即和log与一些构建模型所用的py文件一起保存
    # 注意这里保存的不是只有参数字典model.state_dict()，还有模型结构，即清楚说明了每一层是Conv还是pooling，以及如果是Conv的话其具体配置
    # 这是对每一个epoch都会存一下，但只有accuracy > 0.7 才会存
    if epoch >= push_start and epoch in push_epochs:
    # 这里是在把P project到训练集image中与他最近的patch上，epoch从第十个开始，每十个做一次，但是这里不代表就不做joint training了，而是
    # 比如第十个epoch时，做完joint 再做push，如果设置class_specific=True，则最近的image patch会从与P同属一类的image中去找
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1]) 这个是没有normalize的，见train_push_dataset
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1, # 这个stride可以不是1，比如认为P所表征/与之比较的feature来自于feature map中的一个2x2的区域
            # 而且不是那个滑动窗口每次不是移动1像素,而是可能移动了2个像素，那么stride就是2了 当然这里是1x1
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.70, log=log)
        # 在push完成后也会做一遍test并存一遍，每次push完之后会对FC层做单独的训练，每次做20次，相当于做了20个epoch
        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.70, log=log)
   
logclose()

