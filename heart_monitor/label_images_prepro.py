#combine all training folders into one new folder


import os 
import numpy as np
import matplotlib.pyplot
    
import glob, os, shutil
import csv

main="C:/Users/User/Documents/Astar/8 week ra/work"
dest_dir="C:/Users/User/Documents/Astar/8 week ra/work/spec_images"

folders= glob.iglob(os.path.join(main, "*"))

training_folders=[]
for dirName, subdirList, fileList in os.walk(main):
    for i in subdirList:
        for j in i:
            if(i.startswith('training-')):
                file=os.path.join(main + "/" + i)
                training_folders.append(file)
training_folders=list(set(training_folders))
print(training_folders)
for t in training_folders:
    print(t)
    for dirName, subdirList, fileList in os.walk(t):
        for i in fileList:
            print(i)
            if(i.startswith("REFERENCE")):
                csvname=os.path.join(t + "/" + i)
                print(str(csvname))
                shutil.copy2(csvname, dest_dir)

            
target= "C:/Users/User/Documents/Astar/8 week ra/work/spec_images"
dict_data={}
for dirName, subdirList, fileList in os.walk(target):
    for i in fileList:
        #print(i)
        if(i.startswith("REFERENCE")):
            with open(target+'/'+i) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    filepath=target+'/'+row[0]+'.png'
                    dict_data.update({filepath:row[1]})
                
abnormal="C:/Users/User/Documents/Astar/8 week ra/work/abnormal_normal/abnormal"

normal="C:/Users/User/Documents/Astar/8 week ra/work/abnormal_normal/normal"

for i in range(len(dict_data.keys())):
    if(list(dict_data.values())[i]=='-1'):
        shutil.copy2(list(dict_data.keys())[i], abnormal)
    if(list(dict_data.values())[i]=='1'):
        shutil.copy2(list(dict_data.keys())[i], normal)
        
        




                    
            