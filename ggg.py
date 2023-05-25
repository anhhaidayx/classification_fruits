import PIL
import os
import os.path
from PIL import Image

f = r'C:\\Users\\tienq\\OneDrive\\Desktop\\hoa_qua\\khế'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((224,224))
    img = img.convert('RGB')
    img.save(f_img + '.jpg')
    # img.save(f_img)