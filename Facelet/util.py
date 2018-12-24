"""
模型接口调用
"""
from Facelet_Bank.network.facelet_net import *
from Facelet_Bank.util import test_parse as argparse
from Facelet_Bank.data.testData import untransform
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Facelet_Bank.network.decoder import vgg_decoder
from Facelet_Bank.global_vars import *
import imageio
from Facelet_Bank.util import framework
from Facelet_Bank.network.base_network import VGG
import glob
import os
from Facelet_Bank import test_facelet_net
import numpy as np
import torch
import cv2
import torchvision as tv
import torchvision.transforms as transforms
from Facelet_Bank.data import base_dataset

mean = torch.Tensor((0.485, 0.456, 0.406))
stdv = torch.Tensor((0.229, 0.224, 0.225))

forward_transform = tv.transforms.Compose(
    [transforms.ToTensor(), tv.transforms.Normalize(mean=mean, std=stdv), base_dataset.FitToQuantum()])


def Load_model(style='younger'):
    """
    模型加载接口，返回加载好的模型
    :param style: 美化效果 facehair：胡子 younger: 年轻化
    :return:
    """
    vgg, facelet, decoder = test_facelet_net.test_loader(style)
    return vgg, facelet, decoder

def Test_model(img, x, y, w, h, vgg, facelet, decoder, strength=5):
    """
    模型美化接口
    :param img: 图片数组
    :param x: 人脸框左上角x
    :param y: 人脸框左上角y
    :param w: 人脸框宽度
    :param h: 人脸框高度
    :param vgg: 模型参数
    :param facelet: 模型参数
    :param decoder: 模型参数
    :param strength: 力度，默认为5
    :return:返回修改后的整张图片
    """
    img_clip = img[x:x + w, y:y + w, :]
    image = forward_transform(img_clip)
    image = torch.unsqueeze(image, 0)
    #img_clip = img_clip.astype('float32')
    #image = torch.from_numpy(img_clip)
    image = util.toVariable(image)
    image = image.cuda()
    output = forward(image, vgg, facelet, decoder, strength)
    output = untransform(output.data[0].cpu())
    output = util.center_crop(output, (w, h))
    img[x:x + w, y:y + w, :] = output
    return img


def forward(image, vgg, facelet, decoder, weight):
    vgg_feat = vgg.forward(image)
    w = facelet.forward(vgg_feat)
    vgg_feat_transformed = [vgg_feat_ + weight * w_ for vgg_feat_, w_ in zip(vgg_feat, w)]
    return decoder.forward(vgg_feat_transformed, image)

if __name__ == '__main__':
    vgg, facelet, decoder = Load_model(style='younger')
    img = cv2.imread('./0010_01.jpg')
    img = Test_model(img,0,0,img.shape[0],img.shape[1],vgg,facelet,decoder)
    cv2.imshow('src', img)
    cv2.waitKey(0)