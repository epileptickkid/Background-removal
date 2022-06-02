
import streamlit as st
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import os

st.header("Удаление фона при помощи DeepLabv3 и UNET")
st.write("Выберите изображение из набора данных Carvana, из которого вы хотите удалить фон:")

uploaded_file = st.file_uploader("Выберите изображение...")


def model_deeplabv3(image):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    input_image = image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # создадим mini-batch

    # переместим изображение и модель в GPU , если это возможно:

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    # create a color pallette, selecting a color for each class

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    cleared = remove(r, input_image)
    return(cleared)


def remove(mask,input_image):
    mask = np.array(mask)
    mask = np.expand_dims(mask,2)
    image =  np.array(input_image)
    D = np.concatenate((mask,mask,mask),2)
    D = np.where(D != 0, image , 0)
    return D



class DoubleConv(nn.Module):
    def init(self, in_channels, out_channels):
        super(DoubleConv, self).init()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
class UNET(nn.Module):
    def init(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).init()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Нижняя часть UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Верхняя часть UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def model_unet(input_image):
    IMAGE_HEIGHT = 160  # 1280 изначально
    IMAGE_WIDTH = 240  # 1918 изначально

    model = torch.load("data/u_net")
    model.eval()

    input_image = input_image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # создадим mini-batch
    if torch.cuda.is_available():
        input_batch = input_batch.to('cpu')
        model.to('cpu')

    with torch.no_grad():
        preds = torch.sigmoid(model(input_batch))
        preds = (preds > 0.5).float()

    mask = preds.cpu().numpy().reshape(160, 240)
    plt.imshow(preds.cpu().numpy().reshape(160, 240))
    input_image.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    img = input_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    cleared = remove(mask, img)
    return cleared



if uploaded_file is not None:
    #src_image = load_image(uploaded_file)
    image = Image.open(uploaded_file)
    st.image(uploaded_file, caption='Входное изображение', use_column_width=True)
    # st.write(os.listdir())
    im = model_deeplabv3(image)
    st.image(im, caption='Результат DeepLabv3', use_column_width=True)
    im = model_unet(image)

    st.image(im, caption='Результат Unet', use_column_width=True)
