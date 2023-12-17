import numpy as np
import sys

def confidence_interval_95(data):
    std = np.std(data)
    n = len(data)
    return 1.96*(std/np.sqrt(n))

def func(string):
    ans = ""
    i = 0
    while i < len(string):
        if string[i] == '[':
            while(string[i] != ']'):
                i+=1
            i+=1
            ans += (" "+string[i])
            i+=1
        else:
            ans += string[i]
            i+=1
    return [float(j) for j in ans.split()]

file = open(sys.argv[1],'r')
lines = file.readlines()
file.close()

lines = [line.strip() for line in lines]

results_list = []
temp = []
for i in range(len(lines)):
    if '-' in lines[i] and len(temp) != 0:
        results_list.append(temp)
        temp = []
    elif '-' not in lines[i] and lines[i] != '':
        temp.append(lines[i])

results_list = [results_list[i] for i in range(len(results_list)) if results_list[i] != ['']]



for lines in results_list:
    try:
        c = 0
        print('---------------------')
        while '[' not in lines[c]:
            print(lines[c])
            c+=1
        
        lines = lines[c:]
        
        results = [func(lines[i]) for i in range(len(lines))]
        results = np.array(results)
        print("Train Accuracy:",round(np.mean(results[:,0], axis=0),5), round(np.std(results[:,0]),5))
        print("Confidence Interval 95 Train:",round(confidence_interval_95(results[:,0]),5))
 
        print("Test Accuracy:",round(np.mean(results[:,2], axis=0),5), round(np.std(results[:,2]),5))
        print("Confidence Interval 95:",round(confidence_interval_95(results[:,2]),5))
        print('---------------------')
        print()
    except:
        pass