import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

model_dir = './pretrained_models'

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
} # Vgg模型的结构，但没有涵盖FC层

class VGG_features(nn.Module):

    def __init__(self, cfg, batch_norm=False, init_weights=True):
        super(VGG_features, self).__init__()

        self.batch_norm = batch_norm

        self.kernel_sizes = []
        self.strides = []
        self.paddings = []

        self.features = self._make_layers(cfg, batch_norm) # 构建Vgg模型的各个层

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules(): # 应该是读出了这个模型中属于Module类的每一个子模块，其实Vgg19本身也是一个Module的子模块，因此
            # 会和其内部层一起被读出来，但不会被初始化。而是只有Vgg19内部的层会被初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1) # beta初始化1，gamma初始化0
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, batch_norm):

        self.n_layers = 0

        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

                self.kernel_sizes.append(2)
                self.strides.append(2)
                self.paddings.append(0)

            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]# 这里其实最好不要放inplace = True,
                    # 这个相当于是原地操作，即使x=relu(x),输入的x在原地操作中会在计算图中直接把输入值删去，然后输出赋给输出x，
                    # 那么在反向传播时由于没有输入的输入的值了，就没法求导，可能会报错 Jiabang's alert 在这里我把nn.ReLU(inplace=True)
                    # 改成了nn.ReLU()
                else:
                    layers += [conv2d, nn.ReLU()] #在这里我把nn.ReLU(inplace=True)改成了nn.ReLU()

                self.n_layers += 1

                self.kernel_sizes.append(3)
                self.strides.append(1)
                self.paddings.append(1)

                in_channels = v

        return nn.Sequential(*layers) # *是解包的意思，即把layers这个list中的layer一个个按顺序排列，放到sequential这个容器中,
        # 相当于 nn.sequential(layer1, layer2, ...)

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        '''
        the number of conv layers in the network
        '''
        return self.n_layers

    def __repr__(self):
        template = 'VGG{}, batch_norm={}'
        return template.format(self.num_layers() + 3,
                               self.batch_norm)



def vgg11_features(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['A'], batch_norm=False, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg11'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg11_bn_features(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['A'], batch_norm=True, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg11_bn'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg13_features(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['B'], batch_norm=False, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg13'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg13_bn_features(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['B'], batch_norm=True, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg13_bn'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg16_features(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['D'], batch_norm=False, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg16'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg16_bn_features(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['D'], batch_norm=True, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg16_bn'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


def vgg19_features(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False # 由于在model.py中引用这个函数时只设置了pretrained=True，不传其他关键字参数，因此kwargs是一个
        #空的字典{}，这里相当于是给这个字典里加了一个键值对，相当于是加了一组关键字参数
    model = VGG_features(cfg['E'], batch_norm=False, **kwargs) # 把这个有一个关键字参数的kwargs传入到Vgg类初始化中，相当于在这个类
    # 初始化中，会多加了一个init_weight = False的关键字参数，而在这个类的定义里，会有另一个**kwargs来承接这些自定义的各类关键字参数
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg19'], model_dir=model_dir) #把模型参数文件下载并存到model_dir中
        # （会自动建立这个地址）
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key] # 把与最后分类相关的参数删除，不是值归0，而是就没有classifer相关的参数了
            # 因为这里虽然是transfer learning，但要要适配鸟类分类的类别数量
        model.load_state_dict(my_dict, strict=False) # 这里就是给模型加载权重参数，strict=false的意思就是
        # 如果权重字典里缺少模型中的某些参数（比如 classifier），不会报错。同样，字典中有多余的 key（比如你手动删掉后剩下的），也不会出错。
    return model


def vgg19_bn_features(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG_features(cfg['E'], batch_norm=True, **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['vgg19_bn'], model_dir=model_dir)
        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('classifier'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    return model


if __name__ == '__main__':

    vgg11_f = vgg11_features(pretrained=True)
    print(vgg11_f)

    vgg11_bn_f = vgg11_bn_features(pretrained=True)
    print(vgg11_bn_f)

    vgg13_f = vgg13_features(pretrained=True)
    print(vgg13_f)

    vgg13_bn_f = vgg13_bn_features(pretrained=True)
    print(vgg13_bn_f)

    vgg16_f = vgg16_features(pretrained=True)
    print(vgg16_f)

    vgg16_bn_f = vgg16_bn_features(pretrained=True)
    print(vgg16_bn_f)

    vgg19_f = vgg19_features(pretrained=True)
    print(vgg19_f)

    vgg19_bn_f = vgg19_bn_features(pretrained=True)
    print(vgg19_bn_f)
