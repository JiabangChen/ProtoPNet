import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

from receptive_field import compute_proto_layer_rf_info_v2

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape # prototype_shape=(2000, 512, 1, 1)有2000个P，每个P的H和W是1，D是512
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes) # num_prototypes行，num_classes列
        # 即建立一个独热matrix，来记录每一个P的类别归属，一个P只属于一个类，每个类所包含的P的数量一致，这里相当于是2000/200，一个类有10个P
        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features
        # features 就是一个没有classifier的backbone
        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
            # 即，把backbone中的卷积层提取成一个list，然后找出最后一个卷积层的输出channel数量
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')
            # 意思是只支持以VGG，Resnet，Densenet作为backbone
        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
            # 这里就是backbone输出feature map之后再做两遍1x1的Conv，使得输出的feature map的D与P的D一致，且H W保持不变
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)
        # 初始化prototype vector，且其element的值是在[0,1]之间的
        # 把这个张量包装成一个参数对象；这样它就能在模型中自动注册为 model.parameters() 的一部分；会被优化器自动更新。
        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias
        #最后分类的FC层
        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2 # 这里的x是与P比较的feature map,x的维度是（Batch,512,H,W）
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones) # 这里是在做一个卷积，输入是与P比较的feature map，channel是512，
        # kernel有2000个，相当于最后输出有2000个channel，每个kernel是由512张1x1的矩阵组成的，分别与feature map中的每一层做Conv后相加
        # 且kernel是全1的，相当于是x**2矩阵每一层相加形成一张map，然后这样做了2000个map，这里之所以是2000个map是为了求与2000个P的距离
        #每一个map中的每一个像素就是sum(zi^2),而且是把整张图都都计算出来了，这是为了计算每个z与P的距离，即计算了每一个pixel的sum(zi^2)

        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))# 这里是对2000个P向量逐元素平方之后，沿着维度相加，然后再沿着H和W方向相加，但由于H W都是1
        # 因此最后形成一个2000长度的向量，每一个element相当于是sum(pi^2)
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1) # 这里-1意思是第一个维度让模型自动推断，由于后面是1，1，即加上两个维度，那么第一个维度只能是2000，
        # 因为原本shape是（2000，），所以最后reshape成（2000，1，1）的矩阵

        xp = F.conv2d(input=x, weight=self.prototype_vectors) # 同样的，由于有2000个P，每个P当作一个kernel与feature map做卷积
        # 之所以有两千个P是为了计算feature map与每个P的距离，对于每一个P，其与feature map每一个channel的每一个pixel做卷积就是zi*pi
        # 做完卷积把每个channel加和就形成了sum(zipi),而且这是对整张图做的计算，相当于是计算了每一个pixel，即每一个z的sum(zipi)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast, xp是（BATCH,2000，H，W），而p2_reshape是（2000，1，1）
        # x2_patch_sum and intermediate_result are of the same shape 都是2000张map
        distances = F.relu(x2_patch_sum + intermediate_result)
        # 加 F.relu() 是一种保险措施，防止由于 数值误差 导致距离出现 微小的负值，从而引发逻辑问题或不稳定。
        # 这个计算宗旨是对于feature map中的每一个pixel，即z与2000个P中的每一个P求distance = ||z-p||^2 = sum(zi^2) + sum(pi^2)
        # -2sum(zipi) 之所以把这个距离式子打开算是因为可以一次性用卷积算出所有位置与所有 prototype 的距离，高效并行，
        # 而无需显式构建 z - p 的巨大张量（需要包含feature map中的每一个z与每一个P的距离）。

        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x) # 最后与P比较的feature map
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        distances = self.prototype_distances(x)
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3])) # 用maxpooling找出最小的距离，即最相近的z patch
        min_distances = min_distances.view(-1, self.num_prototypes)
        # 同样这里的view里的-1是自动推断维度，由于maxpooling结果是（B，2000，1，1），然后第二个维度就是2000，那么第一个维度就是B，相当于把
        # 后面两个无用的1省略了
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        # 计算出来的与每个P相似性分数输入FC层，然后输出最后对于200个类别的预测值
        return logits, min_distances

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))
        # 把不需要prune掉的P的索引形成一个列表
        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)
        # 用.data的对self.prototype_vectors进行访问然后形状修改，即只留存prototypes_to_keep中的P，但同时保持另外3个维度，即512，1，1不变
        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0] # P的形状和数量都变了

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes # 最后一层的输入数量变了，但是输出应该还是没变的
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep] # 权重矩阵的形状也变了,这个是直接改了tensor数据

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False) # 在计算feature map与P的距离时，对feature map^2做卷积用到的ones也要改，
        # 为了使卷积输出的channel数量与P一致
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]
        # prototype_class_identity从（2000，200）变成（2000-N，200）

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity) # 矩阵转置
        negative_one_weights_locations = 1 - positive_one_weights_locations # 对每一个pixel用1-

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 初始化在做完backbone feature map后，再做的那两次卷积和BN的参数
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)
        # 初始化FC层的参数，即让与此类相关的P与此类的weight是1，不是此类的P与此类的相连weights是-0.5


def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 512, 1, 1), num_classes=200, # 其实是128维的 详见settings
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained) # 建立了一个backbone，并且把classifer
    # 头扔掉，其他backbone部位的参数用url加载好
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    # 计算做完prototype层后的feature map的感受野信息（rf_info）
    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)
    # 建立prototype模型，包括backbone，让backbone的feature map适配P vector的2个Conv和一个BN，P vector，和FC layer
