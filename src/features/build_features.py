import cv2
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split



def train_test(X,y):
    X_tr,Xtest,y_tr,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_tr,Xtest,y_tr,y_test


def train_val(X,y):
    X_tr,X_val,y_tr,y_val=train_test_split(X,y,test_size=0.2,random_state=42)
    return X_tr,X_val,y_tr,y_val

def save(output_path,data,data_names):
    for i in range(len(data_names)):
        save_path=output_path+'/'+data_names[i]+'.npy'
        np.save(save_path,data[i])
    

def main():
    curr_dir=pathlib.Path().cwd().as_posix()
    X=np.load(curr_dir+'/data/interim/X.npy')
    y=np.load(curr_dir+'/data/interim/y.npy')
    
    output_path=curr_dir+'/data/processed'
    X_tr,X_test,y_tr,y_test=train_test(X,y)
    X_tr,X_val,y_tr,y_val=train_val(X_tr,y_tr)
    data=[X_tr,X_val,X_test,y_tr,y_val,y_test]
    data_names=['X_tr','X_val','X_test','y_tr','y_val','y_test']
    save(output_path,data,data_names)

if __name__ =='__main__':
    main()


    
