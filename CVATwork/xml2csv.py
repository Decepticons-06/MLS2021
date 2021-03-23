import os
import xml.etree.ElementTree as ET
import pandas as pd

os.chdir("/home/sysadm/Documents/machine_learning/face_detection/annotations/vehicles_detection")

tree = ET.parse("track.xml")
root = tree.getroot()

row_data = []
columns = ["label", "frame_no", "xtl", "ytl", "xbr", "ybr", "outside"]

for child in root:
    for track in root.iter('track'):
        if track.attrib['id'] == '0':
            for box in track.iter('box'):
                row_data.append({"label": 'car_1', "frame_no": int(box.attrib['frame']), "xtl": int(float(box.attrib['xtl'])), "ytl": int(float(box.attrib['ytl'])),
                                 "xbr": int(float(box.attrib['xbr'])), "ybr": int(float(box.attrib['ybr'])), "outside": int(box.attrib['outside'])})

        if track.attrib['id'] == '1':
            for box in track.iter('box'):
                row_data.append({"label": 'car_2', "frame_no": int(box.attrib['frame']), "xtl": int(float(box.attrib['xtl'])), "ytl": int(float(box.attrib['ytl'])),
                                 "xbr": int(float(box.attrib['xbr'])), "ybr": int(float(box.attrib['ybr'])), "outside": int(box.attrib['outside'])})

        if track.attrib['id'] == '2':
            for box in track.iter('box'):
                row_data.append({"label": 'bike', "frame_no": int(box.attrib['frame']), "xtl": int(float(box.attrib['xtl'])), "ytl": int(float(box.attrib['ytl'])),
                                 "xbr": int(float(box.attrib['xbr'])), "ybr": int(float(box.attrib['ybr'])), "outside": int(box.attrib['outside'])})

df = pd.DataFrame(row_data, columns = columns)
df.to_csv('track.csv', index = False)
