import os
import gzip
import shutil
import torch
import pickle

# S2T DATA
def save(Data, name):
    with gzip.open(name, 'wb') as handle:
        pickle.dump(Data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def find_all(source_dir, file_extension):
   lista = []
   for item in os.listdir(source_dir):
      source_fn = os.path.join(source_dir, item)
      if os.path.isdir(source_fn):
         lista = lista + find_all(source_fn, file_extension)
      elif item.endswith(file_extension):
         lista.append(os.path.join(source_dir, item).replace("\\", "/"))
   return lista

pts = find_all("./i3d-features-gustavo-2/span=16_stride=2/", ".pt")

# Bloco de Notas com os Nomes dos vídeos e as frases
with open("New_Libraria_Texts.txt", encoding="utf-8") as f:
   lines = f.readlines()
   f.close()


i = 0

annotations = []
for pt in pts:

   for line in lines:

      line = line.split(";")
      name = line[0]
      frase = line[-1].replace("\n", "")

      if name in pt:

         i += 1

         torchs = torch.load(pt)

         name = pt.split("/")[-1].split(".")[0]
         signer = name.split("_")[-1]

         result = []
         for tensor in torchs:
            result.append(tensor[0])
         sign = torch.stack(result)

         print(i, name, signer, frase, len(sign), sign[0].shape)

         annotations.append({"name": name, "signer": signer, "gloss": "blah", "text": frase, "sign": sign})

         break

save(annotations, "test16_nl_usuario_gustavo_2.gzip")
