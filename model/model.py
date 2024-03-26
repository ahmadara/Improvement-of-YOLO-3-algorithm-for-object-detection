#######################
#1.activation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import torch.nn as nn
from colorama import Fore
class Mish(nn.Module):
    def __init__(self):
        super(Mish).__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x

################
# 2. convolotion 

norm_name = {"bn": nn.BatchNorm2d}
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "mish": Mish}


class Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, norm=None, activate=None):
        super(Convolutional, self).__init__()

        self.norm = norm
        self.activate = activate

        self.__conv = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                stride=stride, padding=pad, bias=not norm)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)

        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)

        return x
#################
# 3. resudila block 
class Residual_block(nn.Module):
    def __init__(self, filters_in, filters_out, filters_medium):

        super(Residual_block, self).__init__()
        self.__conv1 = Convolutional(filters_in=filters_in, filters_out=filters_medium, kernel_size=1, stride=1, pad=0,
                                     norm="bn", activate="leaky")
        self.__conv2 = Convolutional(filters_in=filters_medium, filters_out=filters_out, kernel_size=3, stride=1, pad=1,
                                     norm="bn", activate="leaky")

    def forward(self, x):
        r = self.__conv1(x)
        r = self.__conv2(r)
        out = x + r

        return out
#####################
# 4.darknet
class Darknet53(nn.Module):

    def __init__(self):
        super(Darknet53, self).__init__()
        self.__conv = Convolutional(filters_in=3, filters_out=32, kernel_size=3, stride=1, pad=1, norm='bn',
                                    activate='leaky')

        self.__conv_5_0 = Convolutional(filters_in=32, filters_out=64, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_0 = Residual_block(filters_in=64, filters_out=64, filters_medium=32)

        self.__conv_5_1 = Convolutional(filters_in=64, filters_out=128, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_1_0 = Residual_block(filters_in=128, filters_out=128, filters_medium=64)
        self.__rb_5_1_1 = Residual_block(filters_in=128, filters_out=128, filters_medium=64)

        self.__conv_5_2 = Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_2_0 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_1 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_2 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_3 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_4 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_5 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_6 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)
        self.__rb_5_2_7 = Residual_block(filters_in=256, filters_out=256, filters_medium=128)

        self.__conv_5_3 = Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_3_0 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_1 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_2 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_3 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_4 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_5 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_6 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)
        self.__rb_5_3_7 = Residual_block(filters_in=512, filters_out=512, filters_medium=256)

        self.__conv_5_4 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=2, pad=1, norm='bn',
                                        activate='leaky')
        self.__rb_5_4_0 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_1 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_2 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)
        self.__rb_5_4_3 = Residual_block(filters_in=1024, filters_out=1024, filters_medium=512)


    def forward(self, x,is_visulazing =False):
        x = self.__conv(x)

        x0_0 = self.__conv_5_0(x)
        x0_1 = self.__rb_5_0(x0_0)

        x1_0 = self.__conv_5_1(x0_1)
        x1_1 = self.__rb_5_1_0(x1_0)
        x1_2 = self.__rb_5_1_1(x1_1)

        x2_0 = self.__conv_5_2(x1_2)
        x2_1 = self.__rb_5_2_0(x2_0)
        x2_2 = self.__rb_5_2_1(x2_1)
        x2_3 = self.__rb_5_2_2(x2_2)
        x2_4 = self.__rb_5_2_3(x2_3)
        x2_5 = self.__rb_5_2_4(x2_4)
        x2_6 = self.__rb_5_2_5(x2_5)
        x2_7 = self.__rb_5_2_6(x2_6)
        x2_8 = self.__rb_5_2_7(x2_7)  # small

        x3_0 = self.__conv_5_3(x2_8)
        x3_1 = self.__rb_5_3_0(x3_0)
        x3_2 = self.__rb_5_3_1(x3_1)
        x3_3 = self.__rb_5_3_2(x3_2)
        x3_4 = self.__rb_5_3_3(x3_3)
        x3_5 = self.__rb_5_3_4(x3_4)
        x3_6 = self.__rb_5_3_5(x3_5)
        x3_7 = self.__rb_5_3_6(x3_6)
        x3_8 = self.__rb_5_3_7(x3_7)  # medium

        x4_0 = self.__conv_5_4(x3_8)
        x4_1 = self.__rb_5_4_0(x4_0)
        x4_2 = self.__rb_5_4_1(x4_1)
        x4_3 = self.__rb_5_4_2(x4_2)
        x4_4 = self.__rb_5_4_3(x4_3)  # large
        if is_visulazing:
          print(Fore.CYAN,' in darknet neural networks, shapes are as follows: ')
         
          print(Fore.GREEN,'        1th layer output:',x0_0.shape)
          print(Fore.GREEN,'        2th layer output:',x1_0.shape)
          print(Fore.GREEN,'        3th layer output:',x1_1.shape)
          print(Fore.GREEN,'        4th layer output:',x1_2.shape)
          print(Fore.GREEN,'        5th layer output:',x2_0.shape)
          print(Fore.GREEN,'        6th layer output:',x2_1.shape)
          print(Fore.GREEN,'        7th layer output:',x2_2.shape)
          print(Fore.GREEN,'        8th layer output:',x2_3.shape)
          print(Fore.GREEN,'        9th layer output:',x2_4.shape)
          print(Fore.GREEN,'        10th layer output:',x2_5.shape)
          print(Fore.GREEN,'        11th layer output:',x2_6.shape)
          print(Fore.GREEN,'        12th layer output:',x2_7.shape)
          print(Fore.RED,'        first output')
          print(Fore.BLACK,'--------------------')
          print(Fore.GREEN,'        13th layer output:',x2_8.shape)

          print(Fore.GREEN,'        14th layer output:',x3_0.shape)
          print(Fore.GREEN,'        15th layer output:',x3_1.shape)
          print(Fore.GREEN,'        16th layer output:',x3_2.shape)
          print(Fore.GREEN,'        17th layer output:',x3_3.shape)
          print(Fore.GREEN,'        18th layer output:',x3_4.shape)
          print(Fore.GREEN,'        19th layer output:',x3_5.shape)
          print(Fore.GREEN,'        20th layer output:',x3_6.shape)
          print(Fore.GREEN,'        21th layer output:',x3_7.shape)
          print(Fore.RED,'        second output')

          print(Fore.GREEN,'        22th layer output:',x3_8.shape)
          print(Fore.BLACK,'--------------------')
          print(Fore.GREEN,'        23th layer output:',x4_0.shape)
          print(Fore.GREEN,'        24th layer output:',x4_1.shape)
          print(Fore.GREEN,'        25th layer output:',x4_2.shape)
          print(Fore.GREEN,'        26th layer output:',x4_3.shape)
          print(Fore.RED,'third output')
          print(Fore.GREEN,'        27th layer output:',x4_4.shape)

          

        return x2_8, x3_8, x4_4


##########################
#5. yolo fpn 
class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        """
        x1 means previous output; x2 means current output
        """
        out = torch.cat((x2, x1), dim=1)
        return out


class FPN_YOLOV3(nn.Module):
    """
    FPN for yolov3, and is different from original FPN or retinanet' FPN.
    """
    def __init__(self, fileters_in, fileters_out):
        super(FPN_YOLOV3, self).__init__()

        fi_0, fi_1, fi_2 = fileters_in
        fo_0, fo_1, fo_2 = fileters_out

        # large
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1,pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv0_0 = Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv0_1 = Convolutional(filters_in=1024, filters_out=fo_0, kernel_size=1,
                                       stride=1, pad=0)


        self.__conv0 = Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                                      activate="leaky")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route0 = Route()

        # medium
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fi_1+256, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv1_0 = Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv1_1 = Convolutional(filters_in=512, filters_out=fo_1, kernel_size=1,
                                       stride=1, pad=0)


        self.__conv1 = Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                                     activate="leaky")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route1 = Route()

        # small
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fi_2+128, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn",
                          activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn",
                          activate="leaky"),
        )
        self.__conv2_0 = Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1,
                                       pad=1, norm="bn", activate="leaky")
        self.__conv2_1 = Convolutional(filters_in=256, filters_out=fo_2, kernel_size=1,
                                       stride=1, pad=0)

    def forward(self, x0, x1, x2,is_visulazing=False):  # large, medium, small
        # large
        r0 = self.__conv_set_0(x0)
        out0 = self.__conv0_0(r0)
        out0 = self.__conv0_1(out0)

        # medium
        r1 = self.__conv0(r0)
        r1 = self.__upsample0(r1)
        x1 = self.__route0(x1, r1)
        r1 = self.__conv_set_1(x1)
        out1 = self.__conv1_0(r1)
        out1 = self.__conv1_1(out1)

        # small
        r2 = self.__conv1(r1)
        r2 = self.__upsample1(r2)
        x2 = self.__route1(x2, r2)
        r2 = self.__conv_set_2(x2)
        out2 = self.__conv2_0(r2)
        out2 = self.__conv2_1(out2)
        if is_visulazing:
          print(Fore.BLACK,'******************************************************************')
          print(Fore.BLACK,'******************************************************************')
          print(Fore.CYAN,'fpn- large:')
          print(Fore.GREEN,'input_size:',x0.shape)
          print(Fore.GREEN,'1 layer:',r0.shape)
          print(Fore.GREEN,'2 layer:',self.__conv0_0(r0).shape)
          print(Fore.GREEN,'3 layer:',self.__conv0_1(self.__conv0_0(r0)).shape)
          print(Fore.BLACK,'-------------------')
          print(Fore.CYAN,'fpn- med:')
          print(Fore.GREEN,'r1:',r1.shape)
          print(Fore.GREEN,'x1:',x1.shape)
          print(Fore.GREEN,'out1:',out1.shape)
          print(Fore.BLACK,'-------------------')
          print(Fore.CYAN,'fpn- small:')
          print(Fore.GREEN,'r2:',r2.shape)
          print(Fore.GREEN,'x2:',x2.shape)
          print(Fore.GREEN,'out2:',out2.shape)



        return out2, out1, out0  # small, medium, large
#################
#6. yolo head
class Yolo_head(nn.Module):
    def __init__(self, nC, anchors, stride):
        super(Yolo_head, self).__init__()

        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride


    def forward(self, p,is_visulazing = False):
        bs, nG = p.shape[0], p.shape[-1]
        p = p.view(bs, self.__nA, 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)
        if is_visulazing:
          print(Fore.BLACK,'******************************************************************')
          print(Fore.BLACK,'******************************************************************')
          print(' in yolo head part shapes are as follows')
          print(Fore.GREEN,'p shape is: ',p.shape)
          
        p_de = self.__decode(p.clone(),is_visulazing )
        if is_visulazing:
          print(Fore.GREEN,'p_de shape is: ',p_de.shape)
        


        return (p, p_de)


    def __decode(self, p,is_visulazing = False):
        batch_size, output_size = p.shape[:2]

        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)

        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]
        conv_raw_prob = p[:, :, :, :, 5:]

        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)
        if is_visulazing:
          print(Fore.GREEN,'pred_xy shape is: ',pred_xy.shape)
          print(Fore.GREEN,'pred_wh shape is: ',pred_wh.shape)
          print(Fore.GREEN,'pred_xywh shape is: ',pred_xywh.shape)
          print(Fore.GREEN,'pred_conf shape is: ',pred_conf.shape)
          print(Fore.GREEN,'pred_prob shape is: ',pred_prob.shape)
          print(Fore.GREEN,'pred_bbox shape is: ',pred_bbox.shape)


        # return pred_bbox.view(-1, 5 + self.__nC) if not self.training else pred_bbox
        return pred_bbox
###############################################
# 7. yolo main
class Yolov3(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self,MODEL,DATA, init_weights=True):
        super(Yolov3, self).__init__()

        self.__anchors = torch.FloatTensor(MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(MODEL["STRIDES"])
        self.__nC = DATA["NUM"]
        self.__out_channel = MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)

        self.__backnone = Darknet53()
        self.__fpn = FPN_YOLOV3(fileters_in=[1024, 512, 256],
                                fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])

        # small
        self.__head_s = Yolo_head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        # medium  
        self.__head_m = Yolo_head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        # large
        self.__head_l = Yolo_head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

        if init_weights:
            self.__init_weights()


    def forward(self, x,is_visulazing = False):
        out = []

        x_s, x_m, x_l = self.__backnone(x)
        x_s, x_m, x_l = self.__fpn(x_l, x_m, x_s)

        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))
        if is_visulazing:
         
          yolo_backbone = self.__backnone(x,is_visulazing=True)
          x_s, x_m, x_l = yolo_backbone
          yolo_fpn = self.__fpn( x_l, x_m, x_s,is_visulazing=True)
          x_s, x_m, x_l = yolo_fpn
          A = self.__head_s(x_s,is_visulazing=True)
          A = self.__head_m(x_m,is_visulazing=True)
          A = self.__head_m(x_l,is_visulazing=True)
          return yolo_backbone,yolo_fpn

        else:

          if self.training:
              p, p_d = list(zip(*out))
              return p, p_d  # smalll, medium, large
          else:
              p, p_d = list(zip(*out))
              # return p, torch.cat(p_d, 0)
              return p, p_d
        
    def __init_weights(self):

        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                print("initing {}".format(m))


    def load_darknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"

        print("load darknet weights : ", weight_file)

        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1

                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w