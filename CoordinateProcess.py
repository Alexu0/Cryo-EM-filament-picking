#test path /home/wubin/Documents/20180419_NLRP1/relion/ManualPick/Micrograph/3_manualpick.star
import numpy as np
import mrcfile
import cv2
import matplotlib.pyplot as plt
import model
import random
from skimage.morphology import skeletonize_3d
from skimage.transform import probabilistic_hough_line


class topaz_coor_to_star:
    def __init__(self,topaz_path):
        #dictionary key=filename, element1=x, element2=y
        path='/home/wubin/Documents/topaz_test/topaz_coordinate.star'
        self.outpath='/home/wubin/Documents/topaz_test/'
        f=open(path)
        content=f.read()
        self.content=content.split('\n')[6:-1]
        self.dictionary={}
        self.key=[]
        self.description=['','data_ ','','loop_ ','_rlnCoordinateX #1 ','_rlnCoordinateY #2 ']
        self.output=''
        for i in self.content:
            filename = i.split('\t')[1]
            self.key.append(filename)

        self.key = list(set(self.key))

        for key in self.key:
            self.dictionary[key] = []



    def process(self):

        for i in self.content:
            coor_x = filename = i.split('\t')[2]
            coor_y = filename = i.split('\t')[3]
            filename = i.split('\t')[1]
            self.dictionary[filename].append([coor_x, coor_y].join('\t'))

    def output(self):
        for key in self.key:
            outpute_name=self.outpath+key[-4]+'_manualpick.star'




class FilamentCoordinate:
    #read coordinate and extract filament coordinate
    def __init__(self,coor_path,img_path,pixel_size=1.06):
        self.description=['','data_ ','','loop_ ','_rlnCoordinateX #1 ','_rlnCoordinateY #2 ']
        self.index=6
        self.coordinate=[]
        self.constant=78.2
        self.dimension_x = 4096
        self.dimension_y = 4096



        self.pixel_size=pixel_size
        self.star_reader(coor_path)
        self.img_preprocess(img_path)




    def star_reader(self,coor_path):
        f=open(coor_path)
        content=f.read()
        content=content.split('\n')
        content=content[self.index:-2]
        if len(content)%2!=0:
            print ("coordinate number is not even")

        for coordinate in content:
            coor_x=float(coordinate[:12])
            coor_y=float(coordinate[13:])
            coor=[coor_x,coor_y]
            self.coordinate.append(coor)
        self.coordinate = np.array(self.coordinate).reshape(-1, 1, 2).astype(np.int)

    def img_preprocess(self,img_path):
        matrix = mrcfile.open(img_path)
        img = matrix.data
        img = img / self.constant
        img = img.astype(np.uint8)

        self.dimension_x=img.shape[0]
        self.dimension_x = img.shape[0]

        img = img.reshape(self.dimension_x, self.dimension_y, 1)
        img = np.concatenate((img, img, img), 2)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.img=gray

    def draw_marked_area(self):
        for i,c in enumerate(self.coordinate):
            if i%2==0:
                coor_1=tuple(self.coordinate[i].tolist()[0])
                coor_2=tuple(self.coordinate[i+1].tolist()[0])
                cv2.line(self.img,coor_1,coor_2,(0,0,0),50)

        plt.imshow(self.img,'gray')

class RelionStarCoordinate:
    #output coordinate file with relion star format
    def __init__(self,coordinate,output_path='None.star',pixel_size=1.06):
        self.coordinate=coordinate
        self.pixel_size=pixel_size
        self.output_path=output_path

        self.description=['# RELION; version 3.0-beta-2, helical autopicking NTU NISB WB','','data_','','loop_ ','_rlnCoordinateX #1 ','_rlnCoordinateY #2 ']
        self.digits='.000000 '
        self.end='\n \n'
        self.output=''

        self.FielnameOut=output_path[:-4]+'_manualpick.star'


    def StarGenerator(self):
        coortext=[]
        for coor in self.coordinate:
            #coorx=coor[0]
            #coory=coor[1]

            coorx=coor[0][0]
            coory=coor[0][1]

            coorx = str(coorx)
            coory = str(coory)

            b_x = (5 - len(coorx)) * " "
            b_y = (5 - len(coory)) * " "

            line = [b_x + coorx + self.digits + b_y + coory + self.digits]
            coortext.extend(line)

        output=self.description+coortext
        output='\n'.join(output)
        self.output=output+self.end

        f=open(self.FielnameOut,'w')
        f.write(self.output)
        f.close()

class CoordinateGenerator:
    def __init__(self,img_path,pixel_size=1.06,):
        self.img_path=img_path
        self.pixel_size=pixel_size


        self.coordinate=None
        self.thr_marked=None

        self.dimension_x = 4096
        self.dimension_y = 4096


        self.blur_kernal_size=5
        self.adaptive_kernal=85
        self.adaptive_constant=2
        self.morphologyEx_kernal_size=15
        self.perimeter_limit=616
        self.approximation_constant=0.00001
        self.distance_between_coordinate=80

    def find_coordinate(self):
        matrix = mrcfile.open(self.img_path)
        img = matrix.data

        constant = (img.max() - img.min()) / 255
        img = (img - img.min()) / constant

        img = img.astype(np.uint8)
        self.dimension_x = img.shape[0]
        self.dimension_y = img.shape[1]


        img = img.reshape(self.dimension_x, self.dimension_y, 1)
        img = np.concatenate((img, img, img), 2)
        self.gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



        #self.med = cv2.blur(self.gray, (self.blur_kernal_size, self.blur_kernal_size))
        self.med = model.contrast_enhance_mul(self.gray,itr=10,kernal_size=self.blur_kernal_size)
        self.enhance=cv2.equalizeHist(self.med)

        th4 = cv2.adaptiveThreshold(self.enhance, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.adaptive_kernal, self.adaptive_constant)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morphologyEx_kernal_size, self.morphologyEx_kernal_size))
        opening = cv2.morphologyEx(th4, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(th4, cv2.MORPH_CLOSE, kernel)

        reverse = 255 - closing

        self.reverse = reverse

        hierarchy = cv2.findContours(reverse, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        raw_image = hierarchy[0]
        hir = hierarchy[1]

        #select coordinate based on length
        hir_length = model.selection_perimeter(hir,self.perimeter_limit)

        #select coordinate based on ratio
        hir_final = model.selection_ratio(hir_length)

        #smooth coordinate
        hir_p = model.contour_approximation(hir_final,self.approximation_constant)

        # generate the marked area
        thr = cv2.blur(self.gray, (5,5))
        thr[:] = 0
        cv2.fillPoly(thr, hir_p, color=(255, 255, 255))

        # generate the skletion
        skel = skeletonize_3d(thr)

        # get the location of skeleton
        selected_coordinate = model.point_selection(skel, distance=self.distance_between_coordinate)

        self.thr_marked=thr
        self.coordinate=selected_coordinate

    def draw_coordinate_circle(self):

        for i in range(len(self.coordinate)):
            cv2.circle(self.enhance, tuple(self.coordinate[i][0].tolist()), 100, (255, 255, 255), thickness=3, lineType=8, shift=0)

    def main(self):
        self.find_coordinate()
        self.draw_coordinate_circle()
        #plt.imshow(self.enhance,'gray')







class CoordinateGenerator_helical:
    def __init__(self,img_path,pixel_size=1.06,):
        self.img_path=img_path
        self.pixel_size=pixel_size


        self.coordinate=None
        self.thr_marked=None

        self.dimension_x = 4096
        self.dimension_y = 4096


        self.blur_kernal_size=5
        self.adaptive_kernal=85
        self.adaptive_constant=2
        self.morphologyEx_kernal_size=15
        self.perimeter_limit=616
        self.approximation_constant=0.00001
        self.distance_between_coordinate=80

        self.helicla_threshold=10
        self.helicla_minimal_length=200
        self.helicla_line_gap=70

    def find_coordinate(self):
        matrix = mrcfile.open(self.img_path)
        img = matrix.data

        constant = (img.max() - img.min()) / 255
        img = (img - img.min()) / constant

        img = img.astype(np.uint8)
        self.dimension_x = img.shape[0]
        self.dimension_y = img.shape[1]


        img = img.reshape(self.dimension_x, self.dimension_y, 1)
        img = np.concatenate((img, img, img), 2)
        self.gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



        #self.med = cv2.blur(self.gray, (self.blur_kernal_size, self.blur_kernal_size))
        self.med = model.contrast_enhance_mul(self.gray,itr=10,kernal_size=self.blur_kernal_size)
        self.enhance=cv2.equalizeHist(self.med)

        th4 = cv2.adaptiveThreshold(self.enhance, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.adaptive_kernal, self.adaptive_constant)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morphologyEx_kernal_size, self.morphologyEx_kernal_size))
        opening = cv2.morphologyEx(th4, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(th4, cv2.MORPH_CLOSE, kernel)

        reverse = 255 - closing

        self.reverse = reverse

        hierarchy = cv2.findContours(reverse, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        raw_image = hierarchy[0]
        hir = hierarchy[1]

        #select coordinate based on length
        hir_length = model.selection_perimeter(hir,self.perimeter_limit)

        #select coordinate based on ratio
        hir_final = model.selection_ratio(hir_length)

        #smooth coordinate
        hir_p = model.contour_approximation(hir_final,self.approximation_constant)

        # generate the marked area
        thr = cv2.blur(self.gray, (5,5))
        thr[:] = 0
        cv2.fillPoly(thr, hir_p, color=(255, 255, 255))

        # generate the skletion
        skel = skeletonize_3d(thr)

        #generate lines
        lines = probabilistic_hough_line(skel, self.helicla_threshold, self.helicla_minimal_length, self.helicla_line_gap)
        lines = np.array(lines)

        lines_length = map(model.line_length, lines)
        order = np.argsort(lines_length)

        lines = lines[order[::-1]]

        lines_selected = model.line_selection(lines)
        lines_selected_flatten = lines_selected.reshape(-1,1,2)


        # get the location of skeleton
        #selected_coordinate = model.point_selection(skel, distance=self.distance_between_coordinate)

        self.thr_marked=thr
        self.coordinate=lines_selected_flatten
        self.lines=lines_selected

    def draw_coordinate_line(self):

        for line in self.lines:
            try:
                x1, y1, x2, y2 = line[0]
            except:
                x1, y1 = line[0]
                x2, y2 = line[1]
            cv2.line(self.enhance, (x1, y1), (x2, y2), (255, 255, 255), thickness=3)


    def main(self):
        self.find_coordinate()
        self.draw_coordinate_line()
        #plt.imshow(self.enhance,'gray')



class StarMotify:
    #motify .star file
    #path="/run/media/wubin/Seagate Backup Plus Drive1/dose_weighted_coordinate/Images-Disc1_GridSquare_10842512_Data_FoilHole_10848439_Data_10846546_10846547_20180612_0702-165486_aligned_mic_DW_extract.star"
    def __init__(self,coor_path):
        self.path=coor_path
        self.index_label_ID = 33
        self.index_label_AT = 34
        self.index_label_AP = 35
        self.index_label_TL = 36
        self.index_label_AF = 37
        self.index_data_star=38
        self.index_data_end=-3

        self.chang_start = 12
        self.truncate_point=26

        self.digit_length=12
        self.ap_length=6

        self.label_ID="_rlnHelicalTubeID #29 "
        self.label_AT="_rlnAngleTiltPrior #30 "
        self.label_AP="_rlnAnglePsiPrior #31 "
        self.label_TL="_rlnHelicalTrackLength #32 "
        self.label_AF="_rlnAnglePsiFlipRatio #33 "

        self.dic={}

        self.dic["12"] = "_rlnImageName  # 8 "
        self.dic["13"] = "_rlnMicrographName  # 9 "
        self.dic["14"] = "_rlnMagnification  # 10 "
        self.dic["15"] = "_rlnDetectorPixelSize  # 11 "
        self.dic["16"] = "_rlnCtfMaxResolution  # 12 "
        self.dic["17"] = "_rlnCtfFigureOfMerit  # 13 "
        self.dic["18"] = "_rlnVoltage  # 14 "
        self.dic["19"] = "_rlnDefocusU  # 15 "
        self.dic["20"] = "_rlnDefocusV  # 16 "
        self.dic["21"] = "_rlnDefocusAngle  # 17 "
        self.dic["22"] = "_rlnSphericalAberration  # 18 "
        self.dic["23"] = "_rlnCtfBfactor  # 19 "
        self.dic["24"] = "_rlnCtfScalefactor  # 20 "
        self.dic["25"] = "_rlnPhaseShift  # 21 "
        self.dic["26"] = "_rlnAmplitudeContrast  # 22 "


        self.ID_info="           1 "
        self.AT_info="   90.000000 "
        self.AP_info="   -35.41706 "
        self.TL_info="    0.000000 "
        self.AF_info="    0.500000 "

        self.f=open(self.path,'r')
        self.content=self.f.read()
        self.f.close()




    def star_process(self):
        content=self.content
        content=content.split('\n')

        if content[22]==self.label_ID:
            print "processed file"
            return

        content.insert(self.index_label_ID, self.label_ID)
        content.insert(self.index_label_AT, self.label_AT)
        content.insert(self.index_label_AP, self.label_AP)
        content.insert(self.index_label_TL, self.label_TL)
        content.insert(self.index_label_AF, self.label_AF)

        #for i in range(self.chang_start,self.chang_start+15):
        #    content[i] = self.dic[str(i)]

        length=len(content)
        for index,c in enumerate(content):
            if index>=self.index_data_star and index<=length+self.index_data_end:
                order=index-self.index_data_star+1
                len_digi=len(str(order))
                number_space=self.digit_length-len_digi
                self.ID_info=" "*number_space+str(order)+" "

                angle=random.randint(-180,180)
                len_ap=len(str(angle))
                number_space_ap=self.ap_length - len_ap
                self.AP_info=" "*number_space_ap+str(angle)+".00000 "


                insert = self.ID_info + self.AT_info + self.AP_info + self.TL_info + self.AF_info
                content[index] = content[index] + insert
                #content[index]=content[index][:self.truncate_point]+insert+content[index][self.truncate_point:]

        content="\n".join(content)
        self.content=content

    def star_write(self):
        self.f=open(self.path,'w')
        self.f.write(self.content)
        self.f.close()


class particles_to_coordinate:
    #output coordinate file with relion star format
    def __init__(self,coordinate,output_path='None.star',pixel_size=1.06):
        self.coordinate=coordinate
        self.pixel_size=pixel_size
        self.output_path=output_path

        self.description=['# RELION; version 3.0-beta-2, helical autopicking NTU NISB WB','','data_','','loop_ ','_rlnCoordinateX #1 ','_rlnCoordinateY #2 ','_rlnClassNumber #3','_rlnAutopickFigureOfMerit #4','_rlnHelicalTubeID #5','_rlnAngleTiltPrior #6','_rlnAnglePsiPrior #7','_rlnHelicalTrackLength #8','_rlnAnglePsiFlipRatio #9']
        self.digits='.000000 '
        self.space_2='            '
        self.space_3='     '
        self.space_4_v='            '
        self.space_5='    '
        self.space_6_v='     '
        self.space_7='     '
        self.space_8='     '
        self.tilt='90.000000'
        self.length='0.000000'
        self.ratio='0.500000'
        self.end='\n \n'
        self.output=''

        self.FielnameOut=output_path[:-4]+'_manualpick.star'


    def StarGenerator(self):
        coortext=[]
        for i,coor in enumerate(self.coordinate):
            id_n=i+1
            coorx=int(coor[0])
            coory=int(coor[1])

            coorx = str(coorx)
            coory = str(coory)

            b_x = (5 - len(coorx)) * " "
            b_y = (5 - len(coory)) * " "

            #merit=random.randint(5110,1063012)
            #merit=merit*0.000001
            merit=0.499569
            merit=("%.6f" % merit)

            #merit=round(merit,6)
            #merit=str(merit)

            #psi=random.randint(0,179)
            #tail=random.randint(100398,990179)
            #psi=str(psi)+'.'+str(tail)
            psi='175.090398'


            #line = [b_x + coorx + self.digits + b_y + coory + self.digits]
            line = b_x + coorx + self.digits + b_y + coory + self.digits
            line = line+self.space_2[:(len(self.space_2)-len(str(id_n))+1)]+str(id_n)
            line = line+self.space_3+merit
            line = line+self.space_4_v[:(len(self.space_4_v)-len(str(id_n))+1)]+str(id_n)
            line = line+self.space_5+self.tilt
            line = line+self.space_6_v[:(len(self.space_6_v)-len(str(psi))+8)]+psi
            line = line+self.space_7+self.length
            line = line+self.space_8+self.ratio
            line = [line]
            coortext.extend(line)

        output=self.description+coortext
        output='\n'.join(output)
        self.output=output+self.end

        f=open(self.FielnameOut,'w')
        f.write(self.output)
        f.close()



























