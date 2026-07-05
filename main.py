#Importin libraries
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

"""   
my_model= YOLO("yolo26n.pt")

img = cv2.VideoCapture("IMG_6723.MOV")

results=my_model(source="IMG_6723.MOV", stream=True,workers=0)

for result in results :
    print(result.boxes.cls)
"""

class Detector(YOLO):
    def __init__(self,model_path: str, source:object ) :
        super().__init__(model_path)
        self.source = source
        
    
    def limit_detection_zone(self, target_zone:list, show : bool = True, save : bool= False):
        try :
            #Read Image 
            y1,y2,x1,x2= target_zone[0],target_zone[1],target_zone[2],target_zone[3]
            img= cv2.imread(self.source)
            croped_source = img[y1:y2, x1:x2]
        except Exception as e :
            #Read Video 
            cap = cv2.VideoCapture(self.source)
            for frame in cap:
                croped_source = frame[y1:y2, x1:x2 ]
        if croped_source is None:
            raise ValueError("No cropped frame could be produced from the provided source.")
        if show :
            print("image is cropped :", croped_source.shape)
            cv2.imshow("cropped_source", croped_source)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save :
            cv2.imwrite("cropped_image.png",croped_source)



    def delete_buffer(self, buffer_yx: list, show: bool = False, save : bool= False):
        try :
            #Read Image 
            y_buff,x_buff= buffer_yx[0],buffer_yx[1]
            img= cv2.imread(self.source)
            y_len,x_len=img.shape[0],img.shape[1]
            croped_source = img[y_buff:y_len - y_buff, x_buff:x_len - x_buff]
        except Exception as e :
            #Read Video 
            cap = cv2.VideoCapture(self.source)
            for frame in cap:
                croped_source = frame[y_buff:y_len - y_buff, x_buff:x_len - x_buff]
        if croped_source is None:
            raise ValueError("No cropped frame could be produced from the provided source.")
        if show :
            print("image is cropped :", croped_source.shape)
            cv2.imshow("cropped_source", croped_source)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save :
            cv2.imwrite("cropped_image.png",croped_source)


def main():

    targt_zone = [0,200,0,200]
    buffer_yx= [100,100]
    test_image =Detector("yolo26n.pt","bus.png")
    
    test_image.limit_detection_zone(target_zone=targt_zone,show= False,save=True)
    test_image.delete_buffer(buffer_yx= buffer_yx,show = False,save = True)

if __name__ == "__main__" :
    main()
