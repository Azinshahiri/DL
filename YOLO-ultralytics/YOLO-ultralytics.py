#------------------------------
#   Ultralytics:    YOLO
#------------------------------
from ultralytics import YOLO
import time




#------------------------------
#   Train:    traffic signs
#------------------------------
# data_yaml = "/Users/azinshahiri/Desktop/YOLO-ultralytics/traffic-signs/traffic-signs.yaml"
# epochs = 20
# imgsz = 640
# pretrained = YOLO('/Users/azinshahiri/Desktop/YOLO-ultralytics/runs/detect/train-nano-100/weights/best.pt')
# pretrained.train(data=data_yaml,epochs=epochs,imgsz=imgsz)


#------------------------------
#   Predict:    traffic signs
#------------------------------
trained = "/Users/azinshahiri/Desktop/YOLO-ultralytics/runs/detect/train-nano-100/weights/best.pt"
image = "/Users/azinshahiri/Desktop/YOLO-ultralytics/traffic-signs/train/images/00115.jpg"
another = '/Users/azinshahiri/Desktop/YOLO-ultralytics/example-images/Schilderwald1.jpg'
webcam = '0'


model = YOLO(trained)
model.predict(source=webcam,show=True)
# model.predict(source=image,show=True)
time.sleep(5)