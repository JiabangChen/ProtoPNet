import math

def compute_layer_rf_info(layer_filter_size, layer_stride, layer_padding,
                          previous_layer_rf_info):
    n_in = previous_layer_rf_info[0] # input size
    j_in = previous_layer_rf_info[1] # receptive field jump of input layer
    r_in = previous_layer_rf_info[2] # receptive field size of input layer
    start_in = previous_layer_rf_info[3] # center of receptive field of input layer

    if layer_padding == 'SAME':
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if (n_in % layer_stride == 0):
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    elif layer_padding == 'VALID':
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    else:
        # layer_padding is an int that is the amount of padding on one side
        pad = layer_padding * 2
        n_out = math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1
        # n_out: 经过一次Conv/max pooling，输出的feature map的size。
    pL = math.floor(pad/2)

    j_out = j_in * layer_stride
    # 经过一次Conv/max pooling, 输出了一个feature map，然后这个j_out，也就是计算下一个层rf_info的j_in，指的是此次feature map
    # 上的两个相邻像素之间的间隔（为1），在input image中对应着j_out个像素的间隔
    r_out = r_in + (layer_filter_size - 1)*j_in # 经过一次Conv/max pooling, 输出了一个feature map，这个feature map上的一个
    # 像素，在input image中对应着r_out*r_out个像素，即此feature map中的一个像素对应在input image上的感受野大小
    start_out = start_in + ((layer_filter_size - 1)/2 - pL)*j_in # 经过一次Conv/max pooling, 输出了一个feature map，
    # 这个feature map上的第一个像素，即左上角那个像素，对应在input image上的感受野的中心在input image上的位置（只看宽度方向上的位置），
    # 当有padding时，假如padding是1，那么input image的坐标系就是从左边的padding开始（从-1开始），真实的input image左边界是0，然后
    # 往右开始为1，2，3...
    return [n_out, j_out, r_out, start_out]

def compute_rf_protoL_at_spatial_location(img_size, height_index, width_index, protoL_rf_info):
    n = protoL_rf_info[0]
    j = protoL_rf_info[1]
    r = protoL_rf_info[2]
    start = protoL_rf_info[3]
    assert(height_index < n)
    assert(width_index < n)

    center_h = start + (height_index*j)# 当我知道了feature map中的左上角那个pixel对应在input image上的rf的中心点时，我可以根据
    # feature map中每两个相邻中心点之间对应了在input image层面上的相隔像素数量和与P最近的patch在feature map上的位置，来计算出这个patch
    # 对应在input image上的rf的中心点
    center_w = start + (width_index*j)

    rf_start_height_index = max(int(center_h - (r/2)), 0)
    rf_end_height_index = min(int(center_h + (r/2)), img_size)
    # 之所以要加上max和min是由于一开始做Conv有padding的情况，start_out是从负数开始计数的，比如一个r_out=33，但是start_out却不是16.5
    # 而是8.5，因为有paddings的存在，因此比如是从加了padding后的最左边-8开始计数，中心点恰好在8.5上
    rf_start_width_index = max(int(center_w - (r/2)), 0)
    rf_end_width_index = min(int(center_w + (r/2)), img_size)
    # return的结果是与P最近的patch对应在input image上的rf，即它的感受野的位置
    return [rf_start_height_index, rf_end_height_index,
            rf_start_width_index, rf_end_width_index]

def compute_rf_prototype(img_size, prototype_patch_index, protoL_rf_info):
    img_index = prototype_patch_index[0]
    height_index = prototype_patch_index[1]
    width_index = prototype_patch_index[2]
    # 与P最近的patch的位置（batch中第index张输入图中的第H，W位置的patch）
    rf_indices = compute_rf_protoL_at_spatial_location(img_size,
                                                       height_index,
                                                       width_index,
                                                       protoL_rf_info)
    return [img_index, rf_indices[0], rf_indices[1],
            rf_indices[2], rf_indices[3]]

def compute_rf_prototypes(img_size, prototype_patch_indices, protoL_rf_info):
    rf_prototypes = []
    for prototype_patch_index in prototype_patch_indices:
        img_index = prototype_patch_index[0]
        height_index = prototype_patch_index[1]
        width_index = prototype_patch_index[2]
        rf_indices = compute_rf_protoL_at_spatial_location(img_size,
                                                           height_index,
                                                           width_index,
                                                           protoL_rf_info)
        rf_prototypes.append([img_index, rf_indices[0], rf_indices[1],
                              rf_indices[2], rf_indices[3]])
    return rf_prototypes

def compute_proto_layer_rf_info(img_size, cfg, prototype_kernel_size):
    rf_info = [img_size, 1, 1, 0.5]

    for v in cfg:
        if v == 'M':
            rf_info = compute_layer_rf_info(layer_filter_size=2,
                                            layer_stride=2,
                                            layer_padding='SAME',
                                            previous_layer_rf_info=rf_info)
        else:
            rf_info = compute_layer_rf_info(layer_filter_size=3,
                                            layer_stride=1,
                                            layer_padding='SAME',
                                            previous_layer_rf_info=rf_info)

    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
                                                layer_stride=1,
                                                layer_padding='VALID',
                                                previous_layer_rf_info=rf_info)

    return proto_layer_rf_info

def compute_proto_layer_rf_info_v2(img_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size):

    assert(len(layer_filter_sizes) == len(layer_strides)) # assert，断言，如果assert括号内不为true,就抛出异常
    assert(len(layer_filter_sizes) == len(layer_paddings))
    # 这里输入三个列表是backbone这个模型，经过每一层之后，收集到的每一层的kernel size，stride size, padding size。
    rf_info = [img_size, 1, 1, 0.5]

    for i in range(len(layer_filter_sizes)):
        filter_size = layer_filter_sizes[i]
        stride_size = layer_strides[i]
        padding_size = layer_paddings[i]

        rf_info = compute_layer_rf_info(layer_filter_size=filter_size,
                                layer_stride=stride_size,
                                layer_padding=padding_size,
                                previous_layer_rf_info=rf_info)
        # 对backbone中的每一层做receptive field info计算，即计算[n_out, j_out, r_out, start_out]
    # for 循环结束的rf_info结果是对backbone流程走完的feature map的[n_out, j_out, r_out, start_out]
    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
                                                layer_stride=1,
                                                layer_padding='VALID',
                                                previous_layer_rf_info=rf_info)
    # 这个prototype层其实也相当于是backbone中的一层，具体来说，是一个CNN层，只不过不做卷积做了一个L2 log distance 计算而已，因此做完
    # prototype层后的rf_info，即用feature map和P计算完相似度分数之后的map的rf_info / [n_out, j_out, r_out, start_out]，也是
    # 用compute_layer_rf_info计算相似的流程，只不过kernel size是1，stride是1，padding是VALID，然后previous_layer_rf_info是
    # 走完backbone生成的feature map的rf_info
    return proto_layer_rf_info

