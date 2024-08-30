import cv2
import numpy as np
import os
import pathlib
# 0->carcinoma
# 1->keratosis
# 2->acne
# 3->eczema
# 4->rosacea
# 5->milia

def load_data(data_path):
    classes=os.listdir(data_path)
    
    return classes

def interim_data(data_path:str,X:list,y:list,classes:list):
    for i in range(len(classes)):
        condition_type=classes[i]
        imgs=os.listdir(data_path+'/'+condition_type)
        
        for j in imgs:
            img=cv2.imread(data_path+'/'+condition_type+'/'+j)
            img=cv2.resize(img,(224,224))
            X.append(img)
            y.append(i)
    X=np.array(X)
    y=np.array(y)
    return X,y


def save_to_path(output_path,X,y):
    np.save(output_path+'/'+'X.npy',X)
    np.save(output_path+'/'+'y.npy',y)

    
def main():
    curr_dir=pathlib.Path().cwd().as_posix()
    data_path=curr_dir+'/data/raw/Skin_Conditions'
    output_path=curr_dir+'/data/interim' 
    classes=load_data(data_path)
    X=[]
    y=[]
    X,y=interim_data(data_path,X,y,classes)
    save_to_path(output_path,X,y)

          

if __name__ =='__main__':
    main()