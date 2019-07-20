from os import listdir
from os.path import join, isfile, isdir
#import pandas as pd
from darknet import load_net, load_meta, detect

if __name__ == "__main__":
    mypath = "../data/car_count_old_london"

    net = load_net("../cfg/yolov3-spp.cfg","yolov3-spp.weights",0)
    meta = load_meta("../cfg/coco.data")

    cols = ["date","camera_id","image_id","car_count","bus_count","motorbike_count","truck_count","bicycle_count"]

    # df = pd.DataFrame(columns=["date","camera_id","image_id","car_count","bus_count","motorbike_count","truck_count","bicycle_count"])
                               # "car_thresh", "bus_thresh", "motorbike_thresh", "truck_thresh", "bicycle_thresh"])


    with open("count_results.csv",'a') as myfile:
        myfile.write(", ".join(cols) + "\n")



    for date in [d for d in listdir(mypath) if isdir(join(mypath,d))]:
        datepath = join(mypath,date)

        for camera_id in [cid for cid in listdir(datepath) if isdir(join(datepath,cid))]:
            campath = join(datepath,camera_id)

            for image_id in [img for img in listdir(campath) if isfile(join(campath,img)) and img.endswith(".jpg")]:
                imagepath = join(campath,image_id)

                r = detect(net, meta, imagepath, thresh = 0.25)

                car_count = 0
                bus_count = 0
                motorbike_count = 0
                truck_count = 0
                bicycle_count = 0
                for (cls, thresh, bbox) in r:
                    if cls == "car":
                        car_count+=1
                    elif cls == "bus":
                        bus_count+=1
                    elif cls == "motorbike":
                        motorbike_count+=1
                    elif cls == "truck":
                        truck_count+=1
                    elif cls == "bicycle":
                        bicycle_count+=1


                with open("count_results.csv", 'a') as myfile:
                    myfile.write(", ".join([date,camera_id,image_id,str(car_count),str(bus_count),str(motorbike_count),str(truck_count),str(bicycle_count)])+ "\n")

