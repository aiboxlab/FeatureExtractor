
import os
import re
import math
import gzip
import json
import pickle
import utils as u
import pandas as pd
import numpy as np
import torch
import cv2
import datetime as dt
from datetime import datetime
from models.pytorch_i3d import InceptionI3d


#import torch
#print(torch.cuda.is_available())
# exec(open('extract_features_novo_dataset.py').read())


# S2T DATA
def save(Data, name):
    with gzip.open(name, 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_data_with_subdirectorys(data_path, ext='.mp4'):
    videos_path_list = []
    print("List of all directories in '% s':" % data_path)

    for path, subdirs, files in os.walk(data_path):
        for name in files:
            if name.endswith(ext):
                video = os.path.join(path, name).replace("\\","/")
                videos_path_list.append(video)

    return videos_path_list


# RESCALING
def crop_slices(image, height, width, slice_size):
    return image[0:height, 0+slice_size:width-slice_size]

def resize_square(image, resolution_factor):
    return cv2.resize(image, (resolution_factor, resolution_factor)) 

def make_frame(image, to_resolution):
    height, width, _ = image.shape

    #Difereça entre as dimensões de largura e altura (tamanho dopedaço que será cortado)
    diff = max(height, width) - min(height, width)

    #Vamos dividir o "pedaço" anterior em dois pedaços de igual tamanho (para remover 1 da esquerda e outro da direita)
    slice_size= int(math.floor(diff/2))

    #Vamos cropar o "pedaço" esquedo e o "pedaço" direito
    crop_img = crop_slices(image, height, width, slice_size)#image[0:height, 0+slice_size:width-slice_size]
    #image = cv2.rectangle(image, (0+slice_size, 0), (width-slice_size, height), (255, 0, 0), 2) #Proporção blue

    #Após termos uma imagem quadrada, iremos redimensionar
    resize_img = resize_square(crop_img, to_resolution)

    #resize_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

    #print(resize_img.shape)

    return resize_img


# FEATURE EXTRACTOR
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def transform_frame(cap, resize_img, desired_channel_order):

    try:
        ret, frame = cap.read()

        frame = make_frame(frame, resize_img) # Rescaling

        frame = cv2.resize(frame, dsize=(224, 224))

        frame_transformed = frame.copy()   
            
        if desired_channel_order == 'bgr':
            frame_transformed = frame_transformed[:, :, [2, 1, 0]]

        frame_transformed = (frame_transformed / 255.) * 2 - 1

        return frame_transformed
    except:
        a = []
        return a

def load_all_rgb_frames_from_video_rescaling(video, desired_channel_order='rgb'):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    i = 0
    frames = []
    while(True):
        if fps < 30:
            frame_transformed = transform_frame(cap, 224, desired_channel_order)
            if len(frame_transformed) == 0:
                break
            else:
                frames.append(frame_transformed)
        elif i % 2 == 0:
            frame_transformed = transform_frame(cap, 224, desired_channel_order)
            if len(frame_transformed) == 0:
                break
            else:
                frames.append(frame_transformed)
        i += 1

    nframes = np.asarray(frames, dtype=np.float32)
    return nframes


def extract_features_fullvideo(model, inp, framespan, stride):
    rv = []

    indices = list(range(len(inp)))
    groups = []
    for ind in indices:

        if ind % stride == 0:
            groups.append(list(range(ind, ind+framespan)))

    for g in groups:
        frames = inp[g[0]: min(g[-1]+1, inp.shape[0])]

        num_pad = 9 - len(frames)
        if num_pad > 0:
            pad = np.tile(np.expand_dims(frames[-1], axis=0), (num_pad, 1, 1, 1))
            frames = np.concatenate([frames, pad], axis=0)

        frames = frames.transpose([3, 0, 1, 2])

        ft = _extract_features(model, frames)

        rv.append(ft)

    return rv


def _extract_features(model, frames):
    inputs = torch.from_numpy(frames)

    inputs = inputs.unsqueeze(0)

    inputs = inputs.cuda()
    with torch.no_grad():
        ft = model.extract_features(inputs)
    ft = ft.squeeze(-1).squeeze(-1)[0].transpose(0, 1)

    ft = ft.cpu()
    return ft


def extract_features(video, framespan, stride=2, inp_channels='rgb'):

    weight = 'checkpoints/archive/nslt_2000_065538_0.514762.pt'

    frames = load_all_rgb_frames_from_video_rescaling(video, inp_channels)

    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(2000)
    i3d.load_state_dict(torch.load(weight))
    i3d.cuda()
    i3d.train(False)
    
    features = extract_features_fullvideo(i3d, frames, framespan, stride)

    result = []
    for tensor in features:
        result.append(tensor[0])
    sign = torch.stack(result)

    return sign






if __name__ == "__main__":

    train_signers = ["Gisleile", "Marcelo", "Matheus", "Gabriel", "Thaisa", "Andreia"]
    #test_signers = ["Marcelo"]
    #valid_signers = ["Matheus"]

    trainset = []
    for train in train_signers:
        trainset = trainset + read_data_with_subdirectorys("New_Libraria/"+train, ".mp4")
    trainset = [t for t in trainset if not "MACOSX" in t]

    """
    testset = []
    for test in test_signers:
        testset = testset + read_data_with_subdirectorys("New_Libraria/"+test, ".mp4")
    testset = [t for t in testset if not "MACOSX" in t]

    validset = []
    for valid in valid_signers:
        validset = validset + read_data_with_subdirectorys("New_Libraria/"+valid, ".mp4")
    validset = [v for v in validset if not "MACOSX" in v]
    """


    incr = 1
    quant = (len(trainset)) * 3
    #quant = (len(trainset) + len(testset) + len(validset)) * 3

    #for sset, subset in [("train", trainset), ("test", testset), ("valid", validset)]:
    for sset, subset in [("all", trainset)]:

        #print("\n\n\n", sset, "=========================================")

        for span in [8, 12, 16]:

            annotations = []
            for video in subset:

                print(str(incr)+"/"+str(quant), span, video)
                incr += 1

                #if "MACOSX" in video:
                #    continue

                eaf = video[:-3]+"eaf"

                lines = []
                try:
                    with open(eaf, "r", encoding='utf-8') as f:
                        lines = f.readlines()
                        f.close()
                except:
                    continue

                text = []
                for i, line in enumerate(lines):
                    if "<ANNOTATION_VALUE>" in line:
                        text.append(line)

                        if "</ANNOTATION_VALUE>" in line:
                            break

                        for line in lines[i+1:]:
                            text.append(line)
                            if "</ANNOTATION_VALUE>" in line:
                                break
                    else:
                        continue

                    break

                frase = " ".join(text).replace("\n", " ").replace("<ANNOTATION_VALUE>", "").replace("</ANNOTATION_VALUE>", "")
                frase = re.sub(' +', ' ', frase)[1:].upper()

                name = video.split("/")[-1].split(".")[0]
                signer = video.split("/")[1]

                sign = extract_features(video, span, 2, 'rgb')
                
                annotations.append({"name": name, "signer": signer, "gloss": "blah", "text": frase, "sign": sign})

            save(annotations, sset+str(span)+"_novo_dataset.gzip")











# (Frase Duplicada)
# script_v1_historia_01_gravacao_06_gisleile Gisleile NÃO, TAMBÉM NÃO POSSUO TV COM ENTRADA HDMI. NÃO, TAMBÉM NÃO POSSUO TV COM ENTRADA HDMI.