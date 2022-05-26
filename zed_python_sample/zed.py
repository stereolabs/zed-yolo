import sys
import numpy as np
import pyzed.sl as sl
import cv2
import math


def main() :

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

  
    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()
    #=======================================  yolov4  video test et ============================================           
    #======== Yolov4 Tiny ağırlıklarını yüklemektedir ===================
    weightsPath_tiny = "yolov4.weights"
    configPath_tiny = "yolov4.cfg"

    tiny_net = cv2.dnn.readNetFromDarknet(configPath_tiny, weightsPath_tiny)
    tiny_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    tiny_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(tiny_net)
    
 
    
    def YOLOv4_video(pred_image):
        model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
        image_test = cv2.cvtColor(pred_image, cv2.COLOR_RGBA2RGB)
        image = image_test.copy()
        print('image',image.shape)
        confThreshold= 0.6
        nmsThreshold = 0.4
        classes, confidences, boxes = model.detect(image, confThreshold, nmsThreshold)
        
        return classes,confidences,boxes
        
        
    key = ' '
    LABELS = [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'sofa',
    'pottedplant',
    'bed',
    'diningtable',
    'toilet',
    'tvmonitor',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]
    COLORS = [[0, 0, 255]]
    while key != 113 :
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
            # Retrieve the RGBA point cloud in half resolution
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)
            
            # Get and print distance value in mm at the center of the image
            # We measure the distance camera - object using Euclidean distance
            
            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed.get_data()
            #depth_image_ocv = depth_image_zed.get_data()
            classes,confidences,boxes = YOLOv4_video(image_ocv)
            
            for cl,score,(left,top,width,height) in zip(classes,confidences,boxes):
                start_pooint = (int(left),int(top))
                end_point = (int(left+width),int(top+height))
                
                x = int(left + width/2)
                y = int(top + height/2)
                color = COLORS[0]
                img =cv2.rectangle(image_ocv,start_pooint,end_point,color,3)
                img = cv2.circle(img,(x,y),5,[0,0,255],5)
                text = f'{LABELS[cl]}: {score:0.2f}'
                cv2.putText(img,text,(int(left),int(top-7)),cv2.FONT_ITALIC,1,COLORS[0],2 )
                
                x = round(x)
                y = round(y)
                err, point_cloud_value = point_cloud.get_value(x, y)
                distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                    point_cloud_value[1] * point_cloud_value[1] +
                                    point_cloud_value[2] * point_cloud_value[2])

                print("Distance to Camera at (class : {0}, score : {1:0.2f}): distance : {2:0.2f} mm".format(LABELS[cl], score, distance), end="\r")
                cv2.putText(img,"Distance: "+str(round(distance/1000,2))+'m',(int(int(left+width)-180),int(int(top+height)+30)),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
                
                cv2.imshow("Image", img)
                    
            
            #cv2.imshow("Image", image_ocv)
            #cv2.imshow("Depth", depth_image_ocv)
            
            key = cv2.waitKey(1)


    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")

if __name__ == "__main__":
    main()