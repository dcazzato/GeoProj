import torch
from torch.autograd import Variable
import torch.nn as nn
import skimage
import skimage.io as io
from torchvision import transforms
import numpy as np
import scipy.io as scio
import os, sys 
from PIL import Image
from modelNetM import EncoderNet, DecoderNet, ClassNet, EPELoss
import resampling
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--imgPath", type=str, default= './test/distorted/')
parser.add_argument("--savePath", type=str, default= './test/undistorted/')
parser.add_argument("--modelPath", type=str, default= './pretrainedmodel/')
args = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

model_en = EncoderNet([1,1,1,1,2])
model_de = DecoderNet([1,1,1,1,2])
model_class = ClassNet()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_en = nn.DataParallel(model_en)
    model_de = nn.DataParallel(model_de)
    model_class = nn.DataParallel(model_class)

if torch.cuda.is_available():
    model_en = model_en.cuda()
    model_de = model_de.cuda()
    model_class = model_class.cuda()


model_en.load_state_dict(torch.load(args.modelPath+'/model_en.pkl'), strict=False)
model_de.load_state_dict(torch.load(args.modelPath+'/model_de.pkl'), strict=False)
model_class.load_state_dict(torch.load(args.modelPath+'/model_class.pkl'), strict=False)

model_en.eval()
model_de.eval()
model_class.eval()  

testImgPath = args.imgPath
saveFlowPath = args.savePath

correct = 0


for index, file in enumerate(os.listdir(testImgPath)):
    imgPath = os.path.abspath(os.path.join(testImgPath, file))
    disimgs = Image.open(imgPath).convert('RGB')
    disimgs = disimgs.resize((256, 256))

    im_npy = np.asarray(disimgs.resize((256, 256)))
    disimgs = transform(disimgs)
        
    use_GPU = torch.cuda.is_available()
    if use_GPU:
        disimgs = disimgs.cuda()
    disimgs = disimgs.view(1,3,256,256)
    disimgs = Variable(disimgs)
        
    middle = model_en(disimgs)
    flow_output = model_de(middle)
    clas = model_class(middle)
        
    _, predicted = torch.max(clas.data, 1)
    if predicted.cpu().numpy()[0] == index:
        correct += 1

    u = flow_output.data.cpu().numpy()[0][0]
    v = flow_output.data.cpu().numpy()[0][1]

    multi = 1
    resImg, resMsk = resampling.rectification(im_npy, flow_output.data.cpu().numpy()[0]*multi)
    img_out = Image.fromarray(resImg)
    #img_out = img_out.resize((500,600))
    img_out.save(saveFlowPath + file)

