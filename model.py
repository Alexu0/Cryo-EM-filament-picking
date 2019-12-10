import mrcfile
import cv2
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from scipy.interpolate import splprep, splev
from skimage.transform import probabilistic_hough_line
import math

def contourPerimeter(contour):
    perimeter=cv2.arcLength(contour,True)
    return perimeter

def contrast_enhance_mul(img,itr=1,kernal_size=5):
    out=img

    for i in range(itr):
        enhance = cv2.equalizeHist(out)
        out=cv2.blur(enhance, (5,5))

    return out


def denoise(image,threshold=100):
    size=21
    ran=size/2
    new=image.copy(True)
    dimension=new.shape[0]
    position=list(product(range(ran,dimension-ran),range(ran,dimension-ran)))
    hist=[]
    for x,y in position:
        window=new[x-ran:x+ran,y-ran:y+ran]
        hist.append(window.sum())
        if window.sum()<threshold:
            new[x,y]=0
    hist=np.array(hist)
    return new,hist

def smooth_contours(contours):
    smoothened = []
    for i,contour in enumerate(contours):
        x, y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        try:
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
            tck, u = splprep([x, y], u=None, s=1.0, per=1)
            # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
            u_new = np.linspace(u.min(), u.max(), 25)
            # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
            x_new, y_new = splev(u_new, tck, der=0)
            # Convert it back to numpy format for opencv to be able to display it
            res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
            smoothened.append(np.asarray(res_array, dtype=np.int32))
        except:
            #print (i)
            pass
    return smoothened


def selection_perimeter(hir,low_limit=616):
    perimeter__selection=[]
    perimeter_list = []
    for i in hir:
        cnt = i
        M = cv2.moments(cnt)
        perimeter = cv2.arcLength(cnt,True)
        if perimeter >= low_limit:
            perimeter__selection.append(cnt)
            perimeter_list.append(perimeter)
    #print (area_list)
    return perimeter__selection




def selection_size(hir,low_limit=100000,high_limit=1000000):
    size_selection=[]
    area_list = []
    for i in hir:
        cnt = i
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        if area >= low_limit and area<=high_limit:
            size_selection.append(cnt)
            area_list.append(area)
    #print (area_list)
    return size_selection


def selection_ratio(hir,high_limit=0.05):
    ratio_selection=[]
    peri=np.array(map(contourPerimeter,hir))
    area=np.array(map(cv2.contourArea,hir))

    ratio=area/((peri/2)**2)
    for i,c in enumerate(ratio):
        if c<=high_limit:
            ratio_selection.append(hir[i])

    return ratio_selection





def skeleton(thr,iterations=1):

    kernel = np.ones((15, 15), np.uint8)
    thr = cv2.dilate(thr, kernel, iterations)

    size=np.size(thr)
    skel = np.zeros(thr.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
    done = False
    while (not done):
        eroded = cv2.erode(thr, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(thr, temp)
        skel = cv2.bitwise_or(skel, temp)
        thr = eroded.copy()

        zeros = size - cv2.countNonZero(thr)
        if zeros == size:
            done = True
    return skel


def point_selection(skel,distance):
    selected = np.array([0, 0]).reshape(1, 1, 2)
    try:
        coordinate = cv2.findNonZero(skel)
        while coordinate.shape[0] >= 1:
            current = coordinate[0]
            # add the first point
            selected = np.append(selected, current.reshape(1, 1, 2), 0)

            # calculate distance
            relative_position = coordinate - current
            distance_list = relative_position[:, 0, 0] ** 2 + relative_position[:, 0, 1] ** 2
            coordinate = coordinate[distance_list >= distance ** 2]

        selected = selected[1:, :, :]
        return selected
    except:
        selected = np.array([2000, 2000]).reshape(1, 1, 2)
        return selected





def contour_approximation(hir,distance=0.001):
    processed_hir=[]
    for cnt in hir:
        epsilon = distance * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        processed_hir.append(approx)
    return processed_hir


def line_distance(line):
    try:
        x1, y1, x2, y2 = line[0]
    except:
        x1, y1 = line[0]
        x2, y2 = line[1]
    distance=math.sqrt((x2-x1)**2+(y2-y1)**2)
    return distance

def LineGeneration(enhance,origin,threshold=100,line_length=100,line_gap=100):
    image=origin.copy(True)
    lines=probabilistic_hough_line(origin,threshold=threshold,line_length=line_length,line_gap=line_gap)
    #black=cv2.blur(origin,(5,5))
    #black[:]=0
    black=255-enhance
    black=255-black
    #distance=np.array(map(line_length,lines))
    for line in lines:
        try:
            x1, y1, x2, y2 = line[0]
        except:
            x1, y1 = line[0]
            x2, y2 = line[1]

        cv2.line(black, (x1, y1), (x2, y2), (255, 255, 255), thickness=3)

    plt.subplot(121);plt.imshow(image,'gray')
    plt.subplot(122);plt.imshow(black,'gray')
    return


def line_length(line):
    p1=line[0]
    p2=line[1]

    p_relative=p1-p2
    dst=math.sqrt(p_relative[0]**2+p_relative[1]**2)

    return dst


def line_distance_area(contours):
    return cv2.contourArea(contours)



def line_selection(lines,low_limit_area=5000):
    selected = np.array([0, 0, 0, 0]).reshape(1, 2, 2)

    try:
        while len(lines) >= 1:
            current = lines[0]
            # add the first point
            selected = np.append(selected, current.reshape(1, 2, 2), 0)


            # calculate distance
            current_duplicate=np.array([0,0,0,0]*len(lines)).reshape(-1,2,2)
            current_duplicate[:]=current

            contours_matrix=np.concatenate((current_duplicate,lines),1)
            distance=np.array(map(line_distance_area,contours_matrix))

            lines=lines[distance>=low_limit_area]

            #relative_position = coordinate - current
            #distance_list = relative_position[:, 0, 0] ** 2 + relative_position[:, 0, 1] ** 2
            #coordinate = coordinate[distance_list >= distance ** 2]

        selected = selected[1:, :, :]
        return selected
    except:
        selected = np.array([2000, 2000,2010,2010]).reshape(1, 2, 2)
        return selected






def information_extract(path="/home/wubin/Documents/test/reference/particles.star"):
    info_dic={}
    name_list=[]
    start=33
    end=-2
    x=12
    y=24
    name_s=62
    name_e=180
    f=open(path)
    content=f.read()
    content=content.split('\n')
    content=content[start:-2]
    for i in content:
        name_list.append(i[name_s:name_e])

    for name in name_list:
        info_dic[name]=[]

    for i in content:
        c_x=float(i[:12])
        c_y=float(i[13:24])
        coor=[c_x,c_y]
        name=i[name_s:name_e]
        info_dic[name].append(coor)

    return info_dic


class origin():
    def __init__(self):
        self.path='/home/wubin/Documents/20180419_NLRP1/Micrograph/3.mrc'
        matrix = mrcfile.open(self.path)
        img = matrix.data
        img_1 = img / 78.2
        img_1 = img_1.astype(np.uint8)
        img_1 = img_1.reshape(4096, 4096, 1)
        self.img = np.concatenate((img_1, img_1, img_1), 2)
        self.gray= cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
        print ('raw image prepared')

    def reload(self,path):
        self.path=path
        matrix = mrcfile.open(self.path)
        img = matrix.data
        img_1 = img / 78.2
        img_1 = img_1.astype(np.uint8)
        img_1 = img_1.reshape(4096, 4096, 1)
        self.img = np.concatenate((img_1, img_1, img_1), 2)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        print ('reloading completed\nraw image prepared')

    def blur(self,type='general'):
        if type=='general':
            self.med=cv2.blur(self.gray, (5,5))
        elif type=='median':
            self.med = cv2.medianBlur(self.gray, 5)
        elif type=='bilateralFilter':
            self.med = cv2.bilateralFilter(self.gray, 9, 75, 75)
        else:
            print ('invalid input,using general mode')
            self.med = cv2.blur(self.gray, (5, 5))

    def edges_identify(self,t1=40,t2=80,show=0):
        self.edges = cv2.Canny(self.med, t1, t2)
        if show==0:
            pass
        else:
            plt.imshow(self.edges,'gray')

    def HoughTransform(self):
        img_1=self.img
        plt.subplot(121), plt.imshow(self.edges, 'gray')
        plt.xticks([]), plt.yticks([])

        lines = cv2.HoughLines(self.edges, 1, np.pi / 180, 60)
        lines1 = lines[:, 0, :]

        for rho, theta in lines1[:]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 4096 * (-b))
            y1 = int(y0 + 4096 * (a))
            x2 = int(x0 - 4096 * (-b))
            y2 = int(y0 - 4096 * (a))
            cv2.line(img_1, (x1, y1), (x2, y2), (255, 0, 0), 1)

        plt.subplot(122), plt.imshow(img_1, )
        plt.xticks([]), plt.yticks([])

    def quickMode(self):
        self.blur('general')
        self.edges_identify()
        self.HoughTransform()