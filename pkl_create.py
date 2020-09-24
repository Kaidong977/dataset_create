import pickle
import numpy as np
import os
import math

generation_type = 'train' # 'test', 'validation'

noise_path = 'E:/summer/Way_to_PhD/Nonspeech/'
noise_files = os.listdir(noise_path)

libri_path = 'E:/summer/Way_to_PhD/LibriSpeech/'

save_configs = []
save_nums = 10

for save_num in range(save_nums):

    test = {}

    test['room_size'] = [np.random.uniform(3,10), np.random.uniform(3,10), np.random.uniform(2.5,4)]
    test['RT60'] = np.random.uniform(0.1,0.5)

    device_pos = []
    for i in range(4):
        device_pos.append([np.random.uniform(0.05, test['room_size'][0]-0.05),
                           np.random.uniform(0.05, test['room_size'][1]-0.05),
                           np.random.uniform(0.05, test['room_size'][2]-0.05)])

    test['mic_pos_1_per_device'] = [[device_pos[0]], [device_pos[1]], [device_pos[2]], [device_pos[3]]]

    # Two mics per device
    pos2Mic = []
    for i in range(4):
        degree0 = np.random.randint(0,91)*math.pi/180 #alpha
        degree1 = np.random.randint(0,361)*math.pi/180 #beta
        #mic0's relative position
        diffZ = 0.05*math.sin(degree0)
        diffX = 0.05*math.cos(degree0)*math.sin(degree1)
        diffY = 0.05*math.cos(degree0)*math.cos(degree1)
        #mic0 and mic1's position
        pos2Mic.append([[device_pos[i][0]+diffX,
                         device_pos[i][1]+diffY,
                         device_pos[i][2]+diffZ],
                        [device_pos[i][0]-diffX,
                         device_pos[i][1]-diffY,
                         device_pos[i][2]-diffZ]])
    test['mic_pos_2_per_device'] = pos2Mic

    #Three mics per device
    pos3Mic = []
    count = 0
    while count < 4:
        degree0 = np.random.randint(0,181)*math.pi/180 #alpha
        degree1 = np.random.randint(0,241)*math.pi/180  #beta
        degree2 = np.random.randint(0,181)*math.pi/180  #theta
        #mic0's relative position
        diffZ0 = 0.05*math.sin(degree0)
        diffX0 = 0.05*math.cos(degree0)*math.sin(degree1)
        diffY0 = 0.05*math.cos(degree0)*math.cos(degree1)
        # point p, (x,y) is the same as the center of the projection ellipse
        centerX = -diffX0/2
        centerY = -diffY0/2

        k = math.tan(degree2)
        # long axis and short axis of the ellipse 
        b = math.sqrt(3)/2*0.05*math.sin(degree0)
        a = math.sqrt(3)/2*0.05
        
        if b == 0:
            diffX1Pre = 0
            diffX2Pre = 0
        else:
            diffX1Pre = math.sqrt(1/(1/a**2+k**2/b**2)) #x11
            diffX2Pre = diffX1Pre*(-1)  #x21
        diffY1Pre = diffX1Pre*k  #y11
        diffY2Pre = diffX2Pre*k  #y21
        
        diffX1Pre += centerX
        diffX2Pre += centerX
        diffY1Pre += centerY
        diffY2Pre += centerY

        if 0.05*0.05-diffX1Pre**2-diffY1Pre**2 > 0:
            diffZ1 = math.sqrt(0.05*0.05-diffX1Pre**2-diffY1Pre**2)
        else:
            diffZ1 = 0

        if 0.05*0.05-diffX2Pre**2-diffY2Pre**2 > 0:
            diffZ2 = math.sqrt(0.05*0.05-diffX2Pre**2-diffY2Pre**2)
        else:
            diffZ2 = 0
       #mic1's position 
        diffX1 = (diffX1Pre-centerX)*math.cos(math.pi/2-degree1) + (diffY1Pre-centerY)*math.sin(math.pi/2-degree1)+centerX
        diffY1 = (diffY1Pre-centerY)*math.cos(math.pi/2-degree1) - (diffX1Pre-centerX)*math.sin(math.pi/2-degree1)+centerY
       #mic2's position
        diffX2 = (diffX2Pre-centerX)*math.cos(math.pi/2-degree1) + (diffY2Pre-centerY)*math.sin(math.pi/2-degree1)+centerX
        diffY2 = (diffY2Pre-centerY)*math.cos(math.pi/2-degree1) - (diffX2Pre-centerX)*math.sin(math.pi/2-degree1)+centerY

        if abs(diffX1**2+diffY1**2-diffX1Pre**2-diffY1Pre**2) < 1e-5 and \
           abs(diffX2**2+diffY2**2-diffX2Pre**2-diffY2Pre**2) < 1e-5 and \
           abs(math.sqrt(diffX1**2+diffY1**2+diffZ1**2)-0.05) <1e-4 and \
           abs(math.sqrt(diffX2**2+diffY2**2+diffZ2**2)-0.05) <1e-4 and \
           abs((diffX1*diffX2+diffY1*diffY2+diffZ1*diffZ2)/0.05**2) <= 1 and \
           abs((diffX1*diffX2+diffY1*diffY2-diffZ1*diffZ2)/0.05**2) <= 1 and \
           abs((diffX1*diffX0+diffY1*diffY0+diffZ1*diffZ0)/0.05**2) <= 1 and \
           abs((diffX1*diffX0+diffY1*diffY0-diffZ1*diffZ0)/0.05**2) <= 1 and \
           abs((diffX0*diffX2+diffY0*diffY2+diffZ0*diffZ2)/0.05**2) <= 1 and \
           abs((diffX0*diffX2+diffY0*diffY2-diffZ0*diffZ2)/0.05**2) <= 1:

            #judgement of the angle, only leave the angle=120 degree
            flag0 = math.acos((diffX1*diffX0+diffY1*diffY0+diffZ1*diffZ0)/0.05**2)*180/math.pi
            flag1 = math.acos((diffX1*diffX0+diffY1*diffY0-diffZ1*diffZ0)/0.05**2)*180/math.pi
        
            flag2 = math.acos((diffX2*diffX0+diffY2*diffY0+diffZ2*diffZ0)/0.05**2)*180/math.pi
            flag3 = math.acos((diffX2*diffX0+diffY2*diffY0-diffZ2*diffZ0)/0.05**2)*180/math.pi

            flag4 = math.acos((diffX1*diffX2+diffY1*diffY2+diffZ1*diffZ2)/0.05**2)*180/math.pi
            flag5 = math.acos((diffX1*diffX2+diffY1*diffY2-diffZ1*diffZ2)/0.05**2)*180/math.pi
            
            if abs(flag4-120) < 1.:
                if abs(flag0-120) < 1. and abs(flag2-120) < 1.:
                    pos3Mic.append([[device_pos[count][0]+diffX0,
                                 device_pos[count][1]+diffY0,
                                 device_pos[count][2]+diffZ0],
                                [device_pos[count][0]+diffX1,
                                 device_pos[count][1]+diffY1,
                                 device_pos[count][2]+diffZ1],
                                [device_pos[count][0]+diffX2,
                                 device_pos[count][1]+diffY2,
                                 device_pos[count][2]+diffZ2]])
                    count += 1    
                elif abs(flag1-120) < 1. and abs(flag3-120) <1.:
                    pos3Mic.append([[device_pos[count][0]+diffX0,
                                 device_pos[count][1]+diffY0,
                                 device_pos[count][2]+diffZ0],
                                [device_pos[count][0]+diffX1,
                                 device_pos[count][1]+diffY1,
                                 device_pos[count][2]-diffZ1],
                                [device_pos[count][0]+diffX2,
                                 device_pos[count][1]+diffY2,
                                 device_pos[count][2]-diffZ2]])
                    count += 1
            elif abs(flag5-120) < 1.:
                if abs(flag0-120) < 1. and abs(flag3-120) < 1.:
                    pos3Mic.append([[device_pos[count][0]+diffX0,
                                 device_pos[count][1]+diffY0,
                                 device_pos[count][2]+diffZ0],
                                [device_pos[count][0]+diffX1,
                                 device_pos[count][1]+diffY1,
                                 device_pos[count][2]+diffZ1],
                                [device_pos[count][0]+diffX2,
                                 device_pos[count][1]+diffY2,
                                 device_pos[count][2]-diffZ2]])
                    count += 1
                elif abs(flag1-120) < 1. and abs(flag2-120) < 1.:
                    pos3Mic.append([[device_pos[count][0]+diffX0,
                                 device_pos[count][1]+diffY0,
                                 device_pos[count][2]+diffZ0],
                                [device_pos[count][0]+diffX1,
                                 device_pos[count][1]+diffY1,
                                 device_pos[count][2]-diffZ1],
                                [device_pos[count][0]+diffX2,
                                 device_pos[count][1]+diffY2,
                                 device_pos[count][2]+diffZ2]])
                    count += 1

    test['mic_pos_3_per_device'] = pos3Mic

    #four mics per device
    pos4Mic = []
    count = 0
    while count < 4:
        degree0 = np.random.randint(0,91)*math.pi/180
        degree1 = np.random.randint(0,181)*math.pi/180
        degree2 = np.random.randint(0,181)*math.pi/180
        
        diffZ0 = 0.05*math.sin(degree0)
        diffX0 = 0.05*math.cos(degree0)*math.sin(degree1)
        diffY0 = 0.05*math.cos(degree0)*math.cos(degree1)

        k = math.tan(degree2)
        a = 0.05
        b = 0.05*math.sin(degree0)
        if b == 0:
            diffX1Pre = 0
        else:
            diffX1Pre = math.sqrt(1/(1/a**2+k**2/b**2))
        if k < 0:
            diffX1Pre *= -1
        diffY1Pre = diffX1Pre*k
        if 0.05*0.05-diffX1Pre**2-diffY1Pre**2 > 0:
            diffZ1 = math.sqrt(0.05*0.05-diffX1Pre**2-diffY1Pre**2)
        else:
            diffZ1 = 0

        diffX1 = diffX1Pre*math.cos(math.pi/2-degree1) + diffY1Pre*math.sin(math.pi/2-degree1)
        diffY1 = diffY1Pre*math.cos(math.pi/2-degree1) - diffX1Pre*math.sin(math.pi/2-degree1)
        
        if (diffX1*diffX0+diffY1*diffY0+diffZ1*diffZ0)/0.05**2 > 1 or (diffX1*diffX0+diffY1*diffY0+diffZ1*diffZ0)/0.05**2 < -1 \
        or (diffX1*diffX0+diffY1*diffY0-diffZ1*diffZ0)/0.05**2 > 1 or (diffX1*diffX0+diffY1*diffY0-diffZ1*diffZ0)/0.05**2 < -1:
            continue
        else:
            flag0 = math.acos((diffX1*diffX0+diffY1*diffY0+diffZ1*diffZ0)/0.05**2)*180/math.pi
            flag1 = math.acos((diffX1*diffX0+diffY1*diffY0-diffZ1*diffZ0)/0.05**2)*180/math.pi

            if abs(flag0 - 90) < 1.:
                pos4Mic.append([[device_pos[count][0]+diffX0,
                                 device_pos[count][1]+diffY0,
                                 device_pos[count][2]+diffZ0],
                                [device_pos[count][0]-diffX0,
                                 device_pos[count][1]-diffY0,
                                 device_pos[count][2]-diffZ0],
                                [device_pos[count][0]+diffX1,
                                 device_pos[count][1]+diffY1,
                                 device_pos[count][2]+diffZ1],
                                [device_pos[count][0]-diffX1,
                                 device_pos[count][1]-diffY1,
                                 device_pos[count][2]-diffZ1]])
                count += 1
            elif abs(flag1 - 90) < 1.:
                pos4Mic.append([[device_pos[count][0]+diffX0,
                                 device_pos[count][1]+diffY0,
                                 device_pos[count][2]+diffZ0],
                                [device_pos[count][0]-diffX0,
                                 device_pos[count][1]-diffY0,
                                 device_pos[count][2]-diffZ0],
                                [device_pos[count][0]+diffX1,
                                 device_pos[count][1]+diffY1,
                                 device_pos[count][2]-diffZ1],
                                [device_pos[count][0]-diffX1,
                                 device_pos[count][1]-diffY1,
                                 device_pos[count][2]+diffZ1]])
                count += 1

    test['mic_pos_4_per_device'] = pos4Mic


    test['spk_pos'] = [[np.random.uniform(0,test['room_size'][0]),
                        np.random.uniform(0,test['room_size'][1]),
                        np.random.uniform(1.4,2)],
                       [np.random.uniform(0,test['room_size'][0]),
                        np.random.uniform(0,test['room_size'][1]),
                        np.random.uniform(1.4,2)]]
    test['noise_pos'] = [[np.random.uniform(0,test['room_size'][0]),
                          np.random.uniform(0,test['room_size'][1]),
                          np.random.uniform(0,test['room_size'][2])]]
    test['overlap_ratio'] = np.random.uniform(0,1)

    if generation_type == 'train':
        libri_file_first = libri_path+'train-clean-100/'
    elif generation_type == 'test':
        libri_file_first = libri_path+'test-clean/'
    elif generation_type == 'validation':
        libri_file_first = libri_path+'dev-clean/'

    spk0 = libri_file_first
    spk1 = libri_file_first
    libri_file_list = os.listdir(libri_file_first)

    spk0 += libri_file_list[np.random.randint(0,len(libri_file_list))]
    spk0 += '/'
    spk1 += libri_file_list[np.random.randint(0,len(libri_file_list))]
    spk1 += '/'

    libri_file_list = os.listdir(spk0)
    spk0 += libri_file_list[np.random.randint(0,len(libri_file_list))]
    spk0 += '/'
    libri_file_list = os.listdir(spk0)
    spk0 += libri_file_list[np.random.randint(0,len(libri_file_list))]

    libri_file_list = os.listdir(spk1)
    spk1 += libri_file_list[np.random.randint(0,len(libri_file_list))]
    spk1 += '/'
    libri_file_list = os.listdir(spk1)
    spk1 += libri_file_list[np.random.randint(0,len(libri_file_list))]

    test['speech'] = [spk0.split(libri_path)[1], spk1.split(libri_path)[1]]
    test['noise'] = noise_files[np.random.randint(0,len(noise_files))]
    test['start_idx'] = int(np.random.exponential(30000,1)[0])
    test['spk_snr'] = np.random.uniform(0,5)
    test['noise_snr'] = np.random.uniform(10,20)

    save_configs.append(test)

print(save_configs[2])

saving_path = './test_configs.pkl'
with open(saving_path, 'wb') as handle:
	pickle.dump(save_configs, handle)

with open('test_configs.pkl','rb') as f:
    datapos = pickle.load(f)


