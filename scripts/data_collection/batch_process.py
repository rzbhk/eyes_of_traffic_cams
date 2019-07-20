#By Gaurav

# USAGE
# 
# python batch_process.py --input ../data/car_count_old_london --output ../output0 --config ../cfg/yolov3-spp.cfg --weights yolov3-spp.weights --data ../cfg/coco.data --threshold 0.25
# 
# 


import argparse
import cv2
from os import listdir, path, makedirs
from os.path import join, isfile, isdir
from darknet import load_net, load_meta, detect
from tqdm import tqdm

if __name__ == "__main__":
    #parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
        help="path to images for batch processing")
    ap.add_argument("-o", "--output", required=True,
        help="path to output images with bounding boxes")
    ap.add_argument("-c", "--config", required=True,
        help="path to cfg file eg: yolov3-spp.cfg")
    ap.add_argument("-w", "--weights", required=True,
        help="path to weights file eg: yolov3-spp.weights")
    ap.add_argument("-d", "--data", required=True,
        help="path to coco data file eg: coco.data")
    ap.add_argument("-t", "--threshold", type=float, default=0.25,
        help="threshold for the yolo detector")
    args = vars(ap.parse_args())

    # assign the arguments
    inpath = args["input"]
    outpath = args["output"]
    if not path.exists(outpath):
        makedirs(outpath)


    net = load_net(args["config"],args["weights"],0)
    meta = load_meta(args["data"])

    # columns for the csv file
    cols = ["date","camera_id","image_id","car_count","bus_count","motorbike_count","truck_count","bicycle_count"]

    # dataframe which we probably would end up using
    # df = pd.DataFrame(columns=["date","camera_id","image_id","car_count","bus_count","motorbike_count","truck_count","bicycle_count"])
                               # "car_thresh", "bus_thresh", "motorbike_thresh", "truck_thresh", "bicycle_thresh"])


    with open("count_results.csv",'a') as myfile:
        myfile.write(", ".join(cols) + "\n")

    for date in tqdm([d for d in listdir(inpath) if isdir(join(inpath,d))]):
        datepath = join(inpath,date)

        for camera_id in [cid for cid in listdir(datepath) if isdir(join(datepath,cid))]:
            campath = join(datepath,camera_id)
            savedir = join(*[outpath,date,camera_id])
            if not path.exists(savedir):
                makedirs(savedir)

            for image_id in [img for img in listdir(campath) if isfile(join(campath,img)) and img.endswith(".jpg")]:
                imagepath = join(campath,image_id)
                savepath = join(savedir,image_id)
               
                r = detect(net, meta, imagepath, thresh = args["threshold"])
                
                car_count = 0
                bus_count = 0
                motorbike_count = 0
                truck_count = 0
                bicycle_count = 0
                img = cv2.imread(imagepath,cv2.IMREAD_COLOR) #load image in cv2
                (H, W) = img.shape[:2]
                
                for (cls, thresh, bbox) in r:
                    box_color = (0,0,0)
                    if cls == "car":
                        car_count+=1
                        box_color = (0,255,0)
                    elif cls == "bus":
                        bus_count+=1
                        box_color = (225,255,0)
                    elif cls == "motorbike":
                        motorbike_count+=1
                        box_color = (255,0,0)
                    elif cls == "truck":
                        truck_count+=1
                        box_color = (0,0,255)
                    elif cls == "bicycle":
                        box_color = (0,255,255)
                    
                    center_x=int(bbox[0])
                    center_y=int(bbox[1])
                    width = int(bbox[2])
                    height = int(bbox[3])


                    UL_x = int(center_x - width/2) #Upper Left corner X coord
                    UL_y = int(center_y + height/2) #Upper left Y
                    LR_x = int(center_x + width/2)
                    LR_y = int(center_y - height/2)
                    
                        #write bounding box to image
                    cv2.rectangle(img,(UL_x,UL_y),(LR_x,LR_y),box_color,1)
                    #put label on bounding box
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img,cls,(center_x,center_y),font,.5,box_color,1,cv2.LINE_AA)
                cv2.imwrite(savepath,img)
                cv2.waitKey(0) #wait until all the objects are marked and then write out.
                #todo. This will end up being put in the last path that was found if there were multiple
                #it would be good to put it all the paths.


                with open("count_results.csv", 'a') as myfile:
                    myfile.write(", ".join([date,camera_id,image_id,str(car_count),str(bus_count),str(motorbike_count),str(truck_count),str(bicycle_count)])+ "\n")



