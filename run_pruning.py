import os
import shutil

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
# JIABANG'S alert, 要输入完这些才能跑
print(args.gpuid[0])

optimize_last_layer = True

# pruning parameters
k = 6
prune_threshold = 3
# 找出每个P对应的最近的6个image patches，如果这6个image patches中小于三个是来自于P所属的类的，那就认为这个P是表征了背景特征。要删去
# 这里的image patch是未augmented 的训练图，因此做prune是在原图train image上做的
# 模型存储地址和模型名字参照作者注释，这里的模型是已经训练完毕了的，且一定是要刚push完，因为它会用到P的可视，P所对应的那个最近patch的原图
# 等信息（通过epoch和original_model_dir来加载），因此只有push的epoch才能找出相对应的图像文件夹
original_model_dir = args.modeldir[0] #'./saved_models/densenet161/003/'
original_model_name = args.model[0] #'10_16push0.8007.pth'
print(args.modeldir[0])
print(args.model[0])

need_push = ('nopush' in original_model_name)
if need_push:
    assert(False) # pruning must happen after push
    # 如果model name中是nopush，即joint训练后产生的模型，就报错 因为比如第15 16这些epoch是不做push的，因此不会产生P的可视patch等信息
    # 也无法通过epoch去文件夹中调取。
else:
    epoch = original_model_name.split('push')[0]

if '_' in epoch:
    epoch = int(epoch.split('_')[0])
else:
    epoch = int(epoch)
# 找出是第几个epoch产生的model，可以找push后微调FC layer产生的model
model_dir = os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch,
                                          k,
                                          prune_threshold)) # 存放prune后的model以及prune又微调后的model
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir) # 将当前 Python 脚本文件复制到指定目录，并开始记录log

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'prune.log'))

ppnet = torch.load(original_model_dir + original_model_name, weights_only=False) # model加载
# Jiabang's change, plus weights_only=False can load the model successfully
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# load the data
from settings import train_dir, test_dir, train_push_dir

train_batch_size = 80
test_batch_size = 100
img_size = 224
train_push_batch_size = 80

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# train set
train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True)
# Jiabang's change,删去了num_workers=4, pin_memory=False
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False)
# Jiabang's change,删去了num_workers=4, pin_memory=False
log('training set size: {0}'.format(len(train_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# push set: needed for pruning because it is unnormalized
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False) # 这是不做normalization的
# Jiabang's change,删去了num_workers=4, pin_memory=False
log('push set size: {0}'.format(len(train_push_loader.dataset)))

tnt.test(model=ppnet_multi, dataloader=test_loader,
         class_specific=class_specific, log=log) # 先用不做prune的做一遍test，看下精度

# prune prototypes
log('prune')
prune.prune_prototypes(dataloader=train_push_loader,
                       prototype_network_parallel=ppnet_multi,
                       k=k,
                       prune_threshold=prune_threshold,
                       preprocess_input_function=preprocess_input_function, # normalize
                       original_model_dir=original_model_dir,
                       epoch_number=epoch,
                       #model_name=None,
                       log=log,
                       copy_prototype_imgs=True)
accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                class_specific=class_specific, log=log) # 用做过prune的做一遍test，看下精度
save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                            model_name=original_model_name.split('push')[0] + 'prune',
                            accu=accu,
                            target_accu=0.70, log=log) # 存放prune后的model

# last layer optimization
if optimize_last_layer: # prune后可以对FC层做一个微调
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': 1e-4}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    coefs = {
        'crs_ent': 1,
        'clst': 0.8,
        'sep': -0.08,
        'l1': 1e-4,
    }

    log('optimize last layer')
    tnt.last_only(model=ppnet_multi, log=log) # 只学习最后一层的权重，其他的冻结
    for i in range(100):
        log('iteration: \t{0}'.format(i))
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log) # 看下prune后又做了微调后的精度
        save.save_model_w_condition(model=ppnet, model_dir=model_dir,
                                    model_name=original_model_name.split('push')[0] + '_' + str(i) + 'prune',
                                    accu=accu,
                                    target_accu=0.70, log=log) # 把微调后的模型保存，微调会做100次

logclose()
