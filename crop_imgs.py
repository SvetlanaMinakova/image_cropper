import os
from skimage import io,feature
import selectivesearch
import numpy as np
import cv2

class ImgCropper():
    def __init__(self,outpdir='./cropped_imgs/',boxes_num=10,use_Canny=False):
        self.outpdir=outpdir
        self.boxes_num=boxes_num
        self.use_Canny=use_Canny
        self.img_names =[]
        self.boxes=[]

    def crop_all_imgs_from_directoty(self,inpdir):
        self.inpdir=inpdir
        self.get_img_list_from_inp_directory()
        for img in self.img_names:
            (img_path,img_name)=(self.inpdir+'/'+img,self.crop_extension(img))
            print ('cropping image: ',img_name)
            self.get_img_parts(img_path,img_name)

    def crop_image_from_directoty(self,inpdir,img_name):
        self.inpdir=inpdir
        self.get_img_list_from_inp_directory()
        if self.img_names.__contains__(img_name):
            (img_path,img_name)=(self.inpdir+'/'+img_name,self.crop_extension(img_name))
            print ('cropping image: ',img_name)
            self.get_img_parts(img_path,img_name)
            print ('image: ',img_name,' cropped successfully')

    def get_img_list_from_inp_directory(self):
        all_file_names =os.listdir(self.inpdir)
        self.img_names=filter(lambda  x: x.endswith('.jpg'),all_file_names)
        print ('imgs to crop:',self.img_names)

    def crop_extension(self,imgname):
        return (imgname.split('.',1)[0])

    def get_img_parts(self,img_path,img_name):
        self.search_boxes(img_path)
        box_num=0
        for box in self.boxes:
            arrpiece = self.img_as_matrix[box[1]:(box[1]+box[3]),box[0]:(box[0]+box[2]),:]
            io.imsave('./cropped_imgs/'+img_name+str(box_num)+'.jpg',arrpiece)
            box_num+=1

    def search_boxes(self,path):
        self.img_as_matrix= io.imread(path,load_func=self.imread_convert)
        if(self.use_Canny):
            self.apply_Canny()
        self.img_lbl,self.regions = selectivesearch.selective_search(self.img_as_matrix,scale=200,sigma=1.4,min_size=50)
        self.regions.sort(key=lambda  x: x['size'],reverse=True)
        self.boxes = self.get_boxes_witout_duplicates(self.regions)


    def get_boxes_witout_duplicates(self,sorted_regions):
        result_boxes=[]
        if(len(sorted_regions)>0):
            cur_box=(sorted_regions[0])['rect']
            result_boxes.append(cur_box)
            for reg in sorted_regions:
                if(self.boxes_are_different(reg['rect'],cur_box)):
                    result_boxes.append(reg['rect'])
                    cur_box=reg['rect']
                    if(len(result_boxes)>=self.boxes_num):
                        return np.asarray(result_boxes)
        return np.asarray(result_boxes)

    def boxes_are_different(self,box1,box2):
        indexes_num=4 #x1,y1,x2,y2
        for i in range(indexes_num):
            if(box1[i]!=box2[i]):
                return True
        return False

    def apply_Canny(self):
        two_dim_matrix = cv2.Canny(self.img_as_matrix,100,200)
        self.img_as_matrix= self.convert_matrix_to_three_channel_one(two_dim_matrix)
        #feature.canny may be used for 2d input img_array

    def convert_matrix_to_three_channel_one(self,matrix):
        (w,h)=matrix.shape
        three_dim_matrix=[]
        for i in range(w):
            row=[]
            for j in range(h):
                row.append([matrix[i,j],matrix[i,j],matrix[i,j]])
            three_dim_matrix.append(row)
        three_dim_matrix=np.asarray(three_dim_matrix)
        return three_dim_matrix

    #non-float values problem
    def normalize_img_matrix(self,w,h):
        for j in range(h):
            for i in range(w):
                self.img_as_matrix[i,j]/=255
        print (self.img_as_matrix)


    def imread_convert(self,f):
        return io.imread(f).astype(np.float32)

imgs_cropper = ImgCropper(boxes_num=20)
imgs_cropper.crop_all_imgs_from_directoty('./img_examples')
