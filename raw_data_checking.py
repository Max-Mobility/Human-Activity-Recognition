#!/usr/bin/python3

import argparse
import json
from matplotlib.pyplot import plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import pytz
import time
import numpy as np

def main():
    # parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
                    default="./PSDS.json",
                    help="Path to json data file.")
    ap.add_argument("-u", "--uuid", type=str,
                    help="Device UUID to focus on.")
    ap.add_argument("-s", "--start-index", type=int,
                    default=0,
                    help="Start index into array for plotting.")
    ap.add_argument("-e", "--end-index", type=int,
                    default=0,
                    help="End index into array for plotting.")
    args = vars(ap.parse_args())

    fileName = args['input']

    data = []
    with open(fileName) as f:
        data=json.load(f)

    print("Total number of records:", len(data))

    if len(data) == 0:
        exit()

    # determine devices and users in the dataset
    uuids=[]
    userIds=[]
    for i in range(len(data)):
        if 'device_uuid' in data[i].keys():
            uuid = data[i]['device_uuid']
            if uuid not in uuids:
                uuids.append(uuid)
        if 'user_identifier' in data[i].keys():
            userId = data[i]['user_identifier']
            if userId not in userIds:
                userIds.append(userId)

    print("User identifiers: ", userIds)
    print("Device UUIDs: ", uuids)

    # now get the selected data
    deviceUUID = uuids[0]
    if args['uuid']:
        deviceUUID = args['uuid']
    d_data=[]
    d_location=[]
    for i in range(len(data)):
        if 'sensor_data' in data[i].keys():
            if len(data[i]['sensor_data'])>0 and data[i]['device_uuid']==deviceUUID:
                d_data.append(data[i]['sensor_data'])
                #print(len(d_data[-1]),len(d_data)-1,time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data[i]['sensor_data'][1]['t']/1000)))
        if 'location' in data[i].keys() and data[i]['device_uuid']==deviceUUID:
            d_location.append(data[i]['location']);

    print("Number of records for ",deviceUUID,": ",len(d_data))
    print("Number of locations for ",deviceUUID,": ",len(d_location))

    # plot the locations
    long = [x['longitude'] for x in d_location]
    lat = [x['latitude'] for x in d_location]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(long, lat)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

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
    gravity_x=[]
    gravity_y=[]
    gravity_z=[]
    lineAcc_time=[]
    lineAcc_x=[]
    lineAcc_y=[]
    lineAcc_z=[]
    gyro_time=[]
    gyro_x=[]
    gyro_y=[]
    gyro_z=[]
    RV_time=[]
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
    lengths = []
    for dd_data in d_data[startIndex:endIndex]:
        lengths.append(len(dd_data))
        for sensorData in dd_data:
            if sensorData['s'] == 9:
                GR_counter+=1
                gravity_time.append(sensorData['t'] / 1000)
                gravity_x.append(sensorData['d'][0])
                gravity_y.append(sensorData['d'][1])
                gravity_z.append(sensorData['d'][2])
            if sensorData['s'] == 10:
                LA_counter+=1
                lineAcc_time.append(sensorData['t'] / 1000)
                lineAcc_x.append(sensorData['d'][0])
                lineAcc_y.append(sensorData['d'][1])
                lineAcc_z.append(sensorData['d'][2])
            if sensorData['s'] == 4:
                GY_counter +=1
                gyro_time.append(sensorData['t'] / 1000)
                gyro_x.append(sensorData['d'][0])
                gyro_y.append(sensorData['d'][1])
                gyro_z.append(sensorData['d'][2])
            if sensorData['s'] == 15:
                RV_counter +=1
                RV_time.append(sensorData['t'] / 1000)

    if len(d_data[startIndex]) > 1:
        start_time = d_data[startIndex][1]['t']/1000
        end_time = d_data[endIndex-1][1]['t']/1000
        duration = end_time - start_time
    else:
        start_time = d_data[startIndex+1][1]['t']/1000
        end_time = d_data[endIndex-1][1]['t']/1000
        duration = end_time - start_time

    print("Num Sensor Data: ",GR_counter,LA_counter,GY_counter,RV_counter)
    gr_avg, la_avg, gy_avg, rv_avg = [
        duration / GR_counter,
        duration / LA_counter,
        duration / GY_counter,
        duration / RV_counter
    ]
    lenAvg = np.mean(np.array(lengths))
    lenStdDev = np.std(np.array(lengths))
    print("Average sensor interval: {:.4f},{:.4f},{:.4f},{:.4f}".format(gr_avg,la_avg,gy_avg,rv_avg))
    print("Record length (avg, stddev): {:.4f}, {:.4f}".format(lenAvg, lenStdDev))
    print("Start time: ", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(start_time)))
    print("End time:   ", time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(end_time)))
    print("Duration:   {:.2f} seconds".format(duration))
    gravity_time_int=[]
    lineAcc_time_int=[]
    gyro_time_int=[]
    RV_time_int=[]

    gravity_time = np.sort(gravity_time)
    lineAcc_time = np.sort(lineAcc_time)
    gyro_time = np.sort(gyro_time)
    RV_time = np.sort(RV_time)
    '''
    '''
    for i in range (1,GR_counter):
        gravity_time_int.append((gravity_time[i]-gravity_time[i-1]))
    for i in range (LA_counter-1):
        lineAcc_time_int.append((lineAcc_time[i+1]-lineAcc_time[i]))
    for i in range (GY_counter-1):
        gyro_time_int.append((gyro_time[i+1]-gyro_time[i]))
    for i in range (RV_counter-1):
        RV_time_int.append((RV_time[i+1]-RV_time[i]))

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

    '''
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
    '''

    grav_time = mdate.epoch2num(gravity_time)#[x - gravity_time[0]for x in gravity_time])
    gyro_time = mdate.epoch2num(gyro_time)#[x - gyro_time[0]for x in gyro_time])
    fig, ax = plt.subplots()
    ax.plot_date(grav_time, gravity_x, 'y')
    ax.plot_date(gyro_time, gyro_x, 'r')
    # Choose your xtick format string
    date_fmt = '%d-%m-%y %H:%M:%S'

    # Use a DateFormatter to set the data to the correct format.
    date_formatter = mdate.DateFormatter(date_fmt, pytz.timezone('US/Central'))
    ax.xaxis.set_major_formatter(date_formatter)

    # Sets the tick labels diagonal so they fit easier.
    fig.autofmt_xdate()
    plt.show()

if __name__ == "__main__":
    main()
