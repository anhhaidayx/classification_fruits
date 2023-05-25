import numpy as np
from keras_preprocessing import image
import cv2
import os
from keras.models import load_model
vid = cv2.VideoCapture(0)
print("Camera connection successfully established")
i = 0
classes = ['Chuối','dâu tây','dứa','Khế','thanh long','táo']
new_model = load_model('transfer_learning.hdf5')
while(True):
    r, frame = vid.read()
    cv2.imshow('frame', frame)
    cv2.imwrite('C:\\Users\\tienq\\PycharmProjects\\hoa_qua\\image' + str(i) + ".jpg", frame)
    test_image = image.load_img('C:\\Users\\tienq\\PycharmProjects\\hoa_qua\\image'+ str(i) + ".jpg", target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = new_model.predict(test_image)
    result1 = result[0]
    for y in range(6):
        if result1[y] == 1.:
            break
    prediction = classes[y]
    print(prediction)
    os.remove('C:\\Users\\tienq\\PycharmProjects\\hoa_qua\\image' + str(i) + ".jpg")
    i = i + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
# test_image = image.load_img('C:\\Users\\tienq\\OneDrive\\Desktop\\cong-dung-qua-khom.jpg', target_size=(100, 100))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis=0)
# result = new_model.predict(test_image)
# result1 = result[0]
# for y in range(6):
#     if result1[y] == 1.:
#         break
#     prediction = classes[y]
#     print(prediction)
cv2.destroyAllWindows()