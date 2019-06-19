import os
import numpy as np
from pathlib import Path
import pickle

SOURCE = 'raw/HMP_Dataset'
TARGET = 'data/HMP_Dataset'

ActId = {
    'Brush_teeth' : 1,
    'Climb_stairs' : 2,
    'Comb_hair' : 3,
    'Descend_stairs' : 4,
    'Drink_glass' : 5,
    'Eat_meat' : 6,
    'Eat_soup' : 7,
    'Getup_bed':8,
    'Liedown_bed' : 9,
    'Pour_water' : 10,
    'Sitdown_chair' : 11,
    'Standup_chair' : 12,
    'Use_telephone' :13,
    'Walk' : 14}


def parseFile(path):
    data = []
    with open(path,'r') as f:
        for line in f.readlines():
            intLine = [int(x) for x in line.rstrip().split(' ')]
            data.append(intLine)
    return np.array(data)


def getData(activity, source, target, save=True):
    rootPath = os.path.join(source, activity)
    newPath = os.path.join(target, activity)
    if not os.path.isdir(newPath):
        os.mkdir(newPath)
    data = []
    for root, dirs, files in os.walk(rootPath):
        for file in files:
            sourcePath = os.path.join(root, file)
            data.append(parseFile(sourcePath))
    data = np.array(data)
    
    if save:
        with open(os.path.join(newPath, '{}_{}_all.pkl'.format(ActId[activity], activity)), 'wb') as output:
            pickle.dump(data, output)
    return data


def main(source=SOURCE, target=TARGET, save=True):
    for activity in ActId:
        data = getData(activity, source=source, target=target, save=save)
        

if __name__ == '__main__':
    main()