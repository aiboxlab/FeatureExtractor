import cv2
import os
import math
import utils as u



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



to_resolution = 224

dataset = "./New_Libraria"
out_path = "./Compressed_New_Libraria"

if not os.path.exists(out_path):
	os.makedirs(out_path)

lst_videos = u.read_data_with_subdirectorys(dataset, ".mp4")
print("Quantidade de Vídeos: " + str(len(lst_videos)))

lst_videos = list(map(lambda x: x.replace("\\","/") ,lst_videos))

done = os.listdir(out_path)

for video in lst_videos:

	name = video.split("/")[-1]

	if name in done or "MACOSX" in video:
		print("continue")
		continue

	# Read Video
	cap = cv2.VideoCapture(video)
	fps = cap.get(cv2.CAP_PROP_FPS)

	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))
	size = (to_resolution, to_resolution)

	frames = []
	while(cap.isOpened()):
		try:
			ret, frame = cap.read()
			if ret == False:
				break
			frames.append(frame)
		except:
			break

	print(video, fps, size)

	out = cv2.VideoWriter(out_path+"/"+name, cv2.VideoWriter_fourcc(*'MP4V'), 30, (224, 224))

	# Write Video
	for i in range(len(frames)):
		if fps < 30:
			frame_resized = make_frame(frames[i], to_resolution)
			out.write(frame_resized)
		else:
			if i % 2 == 0:
				frame_resized = make_frame(frames[i], to_resolution)
				out.write(frame_resized)

	out.release()





'''
dataset = "./Cesar-Libras-Dataset"
folder = sorted(os.listdir(dataset))

out_path = "./compressed-cesar"


for video in folder:

	if video in os.listdir(out_path):
		continue

	in_video = dataset+"/"+video
	out_video = out_path+"/"+video
	
	cap = cv2.VideoCapture(in_video)
	fps = cap.get(cv2.CAP_PROP_FPS)
	frames = []

	print(in_video, fps)
	while(cap.isOpened()):
		try:
			ret, frame = cap.read()
			if ret == False:
				break
			frames.append(frame)
		except:
			break

	out = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*'MP4V'), 30, (224, 224))

	for i in range(len(frames)):
		if fps < 30:
			img = frames[i][0:1080, 200:1720]
			img = cv2.resize(img, dsize=(224, 224))
			out.write(img)
		else:
			if i % 2 == 0:
				img = frames[i][0:1080, 200:1720]
				img = cv2.resize(img, dsize=(224, 224))
				out.write(img)

	out.release()
'''