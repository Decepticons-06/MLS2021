import sys
import pandas as pd
import glob
import cv2

#reading command line arguments
path_to_csv = sys.argv[1]
path_to_frames = sys.argv[2]

#reading csv file
data = pd.read_csv(path_to_csv+"/track.csv")

#reading the frame paths
path = glob.glob(path_to_frames+"/*.PNG")

#storing the images in df wrt their frame_no
images = pd.DataFrame(columns = ["image"])
start_index = len(path_to_frames+"/frame_") #index of 2 in string "/frame_200.PNG"
for img_file in path:
    frame_no = int(img_file[start_index:start_index+6])
    img_arr = cv2.imread(img_file)
    images.loc[frame_no] = [img_arr]

#range of the annotated frames
min_frame = data["frame_no"].min()
max_frame = data["frame_no"].max()

#drawing bounding boxes
for frame_index in range(min_frame, max_frame+1):
    img = images["image"].loc[frame_index]

    #creating df contains a particular frame's data
    mask = data["frame_no"].values == frame_index
    df = data.loc[mask]

    #creating df contains a particular frame's a particular object's data, this approach helps when a particular frame contains several objects with same label
    mask_car_1 = df["label"].values == "car_1"
    df_car_1 = df.loc[mask_car_1]

    mask_car_2 = df["label"].values == "car_2"
    df_car_2 = df.loc[mask_car_2]

    mask_bike = df["label"].values == "bike"
    df_bike = df.loc[mask_bike]

    #checking whether the object in outside of frame and drawing bounding box
    for index,row in df_car_1.iterrows():
        if df_car_1["outside"].loc[index] == 1:
            continue
        cv2.rectangle(img, (df_car_1["xtl"].loc[index], df_car_1["ytl"].loc[index]), (df_car_1["xbr"].loc[index], df_car_1["ybr"].loc[index]),
                      (255, 0, 0), 1)

    for index,row in df_car_2.iterrows():
        if df_car_2["outside"].loc[index] == 1:
            continue
        cv2.rectangle(img, (df_car_2["xtl"].loc[index], df_car_2["ytl"].loc[index]), (df_car_2["xbr"].loc[index], df_car_2["ybr"].loc[index]),
                      (0, 255, 0), 1)

    for index,row in df_bike.iterrows():
        if df_bike["outside"].loc[index] == 1:
            continue
        cv2.rectangle(img, (df_bike["xtl"].loc[index], df_bike["ytl"].loc[index]), (df_bike["xbr"].loc[index], df_bike["ybr"].loc[index]),
                      (0, 0, 255), 1)
    #showing frame with bounding boxes
    cv2.imshow("frame_{}".format(frame_index), img)
    cv2.waitKey(10)
cv2.destroyAllWindows()
