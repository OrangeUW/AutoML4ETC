import os
import itertools

def printIntersection(directory):
     #prints the side by side intersection of all files in the directory (non recursively)
    
    #Create all possible pairs of files in the given dir
    filePairs = list(itertools.combinations((entry for entry in os.scandir(directory) if entry.is_file()),2))
    

    for f1,f2 in filePairs :
        
        intersection = set.intersection(set(open(f1.__fspath__())), set(open(f2.__fspath__())))
        if(len(intersection)!=0):
            
            print("Intersection between files " + os.path.basename(f1.__fspath__()) + " and " +\
                  os.path.basename(f2.__fspath__()) + " is :")
            print(intersection)
            print("") #Just skips a line for better formatting
            