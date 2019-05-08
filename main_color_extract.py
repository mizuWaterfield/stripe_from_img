# -*- coding: utf-8 -*-

#参考：https://qiita.com/simonritchie/items/396112fb8a10702a3644

import PIL
from PIL import Image, ImageDraw, ImageFont
import cv2
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from numpy import linalg as LA
import random


def draw_random_stripe(color_arr,img_path):
    """
    メインカラーをランダムに選択しランダムな縦幅で矩形描画
    これを繰り返しストライプを作成し，表示し保存
    
    Parameters
    ----------
   color_arr : array
        並べる色の配列
   img_path : str
        抽出元の画像のパス
    
    """
    width = 1024
    height = 1024

    stripe_color_img = Image.new(
        mode='RGB', size=(width, height), color='#333333')
    current_height=0
    while current_height < height:
        random_index = random.randrange(color_arr.shape[0])
        color_hex_str = '#%02x%02x%02x' % tuple(color_arr[random_index])
        random_height = random.randrange(5,70)
        color_img = Image.new(
            mode='RGB', size=(width, random_height),
            color=color_hex_str)
        stripe_color_img.paste(
            im=color_img,
            box=(0, current_height))
        current_height += random_height
    stripe_color_img.show()
    stripe_color_img.save('stripe_'+img_path)

def show_tiled_main_color(color_arr):
    """
    メインカラーを横並びにしたPILの画像を表示する．
    Parameters
    ----------
   color_arr : array
        並べる色の配列
    
    """
    IMG_SIZE = 64
    MARGIN = 15
    width = IMG_SIZE * color_arr.shape[0] + MARGIN * 2
    height = IMG_SIZE + MARGIN * 2

    tiled_color_img = Image.new(
        mode='RGB', size=(width, height), color='#333333')

    for i, rgb_arr in enumerate(color_arr):
        color_hex_str = '#%02x%02x%02x' % tuple(rgb_arr)
        color_img = Image.new(
            mode='RGB', size=(IMG_SIZE, IMG_SIZE),
            color=color_hex_str)
        tiled_color_img.paste(
            im=color_img,
            box=(MARGIN + IMG_SIZE * i, MARGIN))
    tiled_color_img.show()

def extract_main_color(img_path,color_num):
    """
    対象の画像のメインカラーを算出する。

    Parameters
    ----------
    img_path : str
        対象の画像のパス

    Returns
    -------
    cluster_centers_arr : array
        抽出された色の配列
    """
    cv2_img = cv2.imread(img_path)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    cv2_img = cv2_img.reshape(
        (cv2_img.shape[0] * cv2_img.shape[1], 3))

    cluster = KMeans(n_clusters=color_num)
    cluster.fit(X=cv2_img)
    cluster_centers_arr = cluster.cluster_centers_.astype(
        int, copy=False)
    #左上の1ピクセルの色を透過色とみなす
    trans_color = cv2_img[0]
    #透過色(に近い色)を無視する（閾値は適当）
    cluster_centers_arr = np.array([i for i in cluster_centers_arr if LA.norm(np.array(i-trans_color),2) >50 ])
    print("extracted colors array:")
    print(cluster_centers_arr)
    return cluster_centers_arr


img_path = 'your img path'
#第二引数はクラスタの数なので適宜調整
color_arr = extract_main_color(img_path,7)
show_tiled_main_color(color_arr)
draw_random_stripe(color_arr,img_path)
