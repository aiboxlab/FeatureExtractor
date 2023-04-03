import os
import cv2
import torch
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime
from models.pytorch_i3d import InceptionI3d

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def load_all_rgb_frames_from_video(video, desired_channel_order='rgb'):
    cap = cv2.VideoCapture(video)

    frames = []
    faces = []
    while(True):

        frame = np.zeros((224,224,3), np.uint8)

        try:
            ret, frame = cap.read()
            frame = cv2.resize(frame, dsize=(224, 224))

            frame_transformed = frame.copy()   

            if desired_channel_order == 'bgr':
                frame_transformed = frame_transformed[:, :, [2, 1, 0]]

            frame_transformed = (frame_transformed / 255.) * 2 - 1
            frames.append(frame_transformed)

        except:
            break

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
        # numpy array indexing will deal out-of-index case and return only till last available element
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

import os



def listar_arquivos(caminho_pasta, extensao_arquivo):
    arquivos = []
    for pasta_raiz, subpastas, nome_arquivos in os.walk(caminho_pasta):
        for nome_arquivo in nome_arquivos:
            if nome_arquivo.endswith(extensao_arquivo):
                caminho_absoluto = os.path.join(pasta_raiz, nome_arquivo)
                caminho_absoluto = caminho_absoluto.replace("\\", "/")
                arquivos.append(caminho_absoluto)
    return arquivos



def run(weight, frame_roots, outroot, inp_channels='rgb'):
    videos = []

    for root in frame_roots:
        paths = sorted(os.listdir(root))
        videos.extend([os.path.join(root, path) for path in paths])

    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(2000)
    i3d.load_state_dict(torch.load(weight)) # Network's Weight
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    total = 0
    print('feature extraction starts.')

    # ===== extract features ======
    annotations = []
    for framespan, stride in [(16, 2), (12, 2), (8, 2)]:

        outdir = os.path.join(outroot, 'span={}_stride={}'.format(framespan, stride))

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for ind, video in enumerate(videos):
            total += 1
            frames = load_all_rgb_frames_from_video(video, inp_channels)
            features = extract_features_fullvideo(i3d, frames, framespan, stride)
            #annotations.append({"name": name, "signer": signer, "gloss": "---", "text": frase, "sign": sign})
            print(ind, video, len(features))


if __name__ == "__main__":
    weight = 'checkpoints/archive/nslt_2000_065538_0.514762.pt'
    videos_roots = ""
    out = '../i3d-features'
    run(weight, videos_roots, out, 'rgb')
