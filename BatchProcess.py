import os
import CoordinateProcess
from time import time

def main():

    dictionary='/run/media/wubin/Seagate Backup Plus Drive2/luca_process/relion/dose-weighted'
    dirlist=os.listdir(dictionary)
    start_time=time()
    for i,f in enumerate(dirlist):
        if f[-3:]=='mrc':
            path=dictionary+'/'+f

            #process=CoordinateProcess.CoordinateGenerator(path,1.06)
            process=CoordinateProcess.CoordinateGenerator_helical(path,1.06)
            process.main()

            output=CoordinateProcess.RelionStarCoordinate(process.coordinate,path)
            output.StarGenerator()

        if i%20==0:
            current_time=time()
            print ('Number: '+str(i))
            print (current_time-start_time)
            print ('\n')


def star_motify():
    dictionary="/run/media/wubin/Seagate Backup Plus Drive1/dose_weighted_coordinate"
    dirlist = os.listdir(dictionary)
    start_time = time()

    for i,f in enumerate(dirlist):
        if f[-4:]=='star':
            path=dictionary+'/'+f

            process=CoordinateProcess.StarMotify(path)
            process.star_process()
            process.star_write()


        if i%100==0:
            current_time=time()
            print ('Number: '+str(i))
            print (current_time-start_time)



main()
#star_motify()