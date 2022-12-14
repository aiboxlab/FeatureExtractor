import os
import shutil

def selective_copy(source_dir, target_dir, file_extension):
   if not os.path.exists(target_dir):
      os.makedirs(target_dir)

   for item in os.listdir(source_dir):
      source_fn = os.path.join(source_dir, item)
      if os.path.isdir(source_fn):
         selective_copy(source_fn, os.path.join(target_dir, item), file_extension)
      elif item.endswith(file_extension):
         shutil.copyfile(source_fn, os.path.join(target_dir, item))
         print(item)

selective_copy("./New_Libraria_Usuario", "NL_Frases", ".eaf")