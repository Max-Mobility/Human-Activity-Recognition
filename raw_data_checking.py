#!/usr/bin/python3

import argparse
import json
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import time
import numpy as np

def main():
    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
                    default="./PSDS.json",
                    help="Path to json data file.")
    ap.add_argument("-u", "--uuid", type=str,
                    default="bf7ff952e4163201",
                    help="Device UUID to focus on.")
    ap.add_argument("-s", "--start-index", type=int,
                    default=0,
                    help="Start index into array for plotting.")
    ap.add_argument("-e", "--end-index", type=int,
                    default=0,
                    help="End index into array for plotting.")
    args = vars(ap.parse_args())

    fileName = args['input']
    deviceUUID = args['uuid']

    data = []
    with open(fileName) as f:
        data=json.load(f)

    print("Total number of records:", len(data))

    if len(data) == 0:
        exit()

    d_data=[]
    for i in range(len(data)):
        if 'sensor_data' in data[i].keys():
            if len(data[i]['sensor_data'])>0 and data[i]['device_uuid']==deviceUUID:
                d_data.append(data[i]['sensor_data'])
                #print(len(d_data[-1]),len(d_data)-1,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data[i]['sensor_data'][1]['t']/1000)))

    print("Number of records for ",deviceUUID,": ",len(d_data))

    for i in range(len(data)):
        sensors=[]
        if 'sensor_data' in data[i].keys():
            dd_data=data[i]['sensor_data']
            #print(dd_data)
            for sensorData in dd_data:
                if sensorData['s'] not in sensors:
                    sensors.append(sensorData['s'])
            '''
            if len(dd_data)>0:
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dd_data[0]['t']/1000)))
                print(data[i]["device_uuid"]+"  "+data[i]['device_model'] +"  "+data[i]['user_identifier'])
                print(len(dd_data))
                print("sensors:"+str(sensors))
            else:
                print('0 data')
            '''

    gravity_time=[]
    gravity_timestamp=[]
    gravity_x=[]
    gravity_y=[]
    gravity_z=[]
    lineAcc_time=[]
    lineAcc_timestamp=[]
    lineAcc_x=[]
    lineAcc_y=[]
    lineAcc_z=[]
    gyro_time=[]
    gyro_timestamp=[]
    gyro_x=[]
    gyro_y=[]
    gyro_z=[]
    RV_time=[]
    RV_timestamp=[]
    RV_x=[]
    RV_y=[]
    RV_z=[]
    GR_counter=0
    LA_counter=0
    MF_counter=0
    RV_counter=0
    HR_counter=0
    GY_counter=0
    startIndex = 0
    endIndex = len(d_data)
    if args['start_index']:
        startIndex = args['start_index']
    if args['end_index']:
        endIndex = args['end_index']
    for dd_data in d_data[startIndex:endIndex]:

        for sensorData in dd_data:
            if sensorData['s'] == 9:
                GR_counter+=1
                gravity_time.append(sensorData['t'])
                gravity_timestamp.append(sensorData['t']/1000)
                gravity_x.append(sensorData['d'][0])
                gravity_y.append(sensorData['d'][1])
                gravity_z.append(sensorData['d'][2])
            if sensorData['s'] == 10:
                LA_counter+=1
                lineAcc_time.append(sensorData['t'])
                lineAcc_timestamp.append(sensorData['t']/1000)
            if sensorData['s'] == 4:
                GY_counter +=1
                gyro_time.append(sensorData['t'])
                gyro_timestamp.append(sensorData['t']/1000)
            if sensorData['s'] == 15:
                RV_counter +=1
                RV_time.append(sensorData['t'])
                RV_timestamp.append(sensorData['t']/1000)
    print("Num Sensor Data: ",GR_counter,LA_counter,GY_counter,RV_counter)

    start_time = d_data[startIndex][1]['t']/1000
    end_time = d_data[endIndex-1][1]['t']/1000
    print("Start time: ", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(start_time)))
    print("End time:   ", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(end_time)))
    gravity_time_int=[]
    lineAcc_time_int=[]
    gyro_time_int=[]
    RV_time_int=[]
    gravity_timestamp_int=[]
    lineAcc_timestamp_int=[]
    gyro_timestamp_int=[]
    RV_timestamp_int=[]
    '''
    gravity_time = np.sort(gravity_time)
    gravity_timestamp = np.sort(gravity_timestamp)
    lineAcc_time = np.sort(lineAcc_time)
    lineAcc_timestamp = np.sort(lineAcc_timestamp)
    gyro_time = np.sort(gyro_time)
    gyro_timestamp = np.sort(gyro_timestamp)
    RV_time = np.sort(RV_time)
    RV_timestamp = np.sort(RV_timestamp)
    '''
    for i in range (1,GR_counter):
        gravity_time_int.append((gravity_time[i]-gravity_time[i-1]))#/1000000000)
        gravity_timestamp_int.append((gravity_timestamp[i]-gravity_timestamp[i-1])/1000000000)
    for i in range (LA_counter-1):
        lineAcc_time_int.append((lineAcc_time[i+1]-lineAcc_time[i]))#/1000000000)
        lineAcc_timestamp_int.append((lineAcc_timestamp[i+1]-lineAcc_timestamp[i])/1000000000)
    for i in range (GY_counter-1):
        gyro_time_int.append((gyro_time[i+1]-gyro_time[i]))#/1000000000)
        gyro_timestamp_int.append((gyro_timestamp[i+1]-gyro_timestamp[i])/1000000000)
    for i in range (RV_counter-1):
        RV_time_int.append((RV_time[i+1]-RV_time[i]))#/1000000000)
        RV_timestamp_int.append((RV_timestamp[i+1]-RV_timestamp[i])/1000000000)

    '''
    print("gravity time "+str(sum(gravity_time_int)))
    print("gravity timestamp "+str(sum(gravity_timestamp_int)))
    print("lineAcc time "+str(sum(lineAcc_time_int)))
    print("lineAcc timestamp "+str(sum(lineAcc_timestamp_int)))
    print("gyro time "+str(sum(gyro_time_int)))
    print("gyro timestamp "+str(sum(gyro_timestamp_int)))
    print("RV time "+str(sum(RV_time_int)))
    print("RV timestamp "+str(sum(RV_timestamp_int)))
    '''

    fig=plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plot(RV_time_int,'y')
    plot(gravity_time_int,'r')
    plt.subplot(1,2,2)
    plot(lineAcc_time_int,'b')
    plot(gyro_time_int,'k')
    plt.show()

    fig=plt.figure(figsize=(10,5))
    start_index=0#71000
    end_index = -1#GY_counter#150000
    plt.subplot(1,2,1)
    plot(RV_timestamp_int[start_index:end_index],'y')
    plot(gravity_timestamp_int[start_index:end_index],'r')
    plt.subplot(1,2,2)
    plot(lineAcc_timestamp_int[start_index:end_index],'b')
    plot(gyro_timestamp_int[start_index:end_index],'k')
    plt.show()

    plot([x - gravity_time[0]for x in gravity_time])#,gravity_x)
    plt.show()

    plot((gyro_timestamp))
    plt.show()

    plot((gravity_timestamp))
    plt.show()

if __name__ == "__main__":
    main()
