from keras.src.applications import resnet
from keras.src.models import Model
from keras.src.layers import Dense,GlobalAveragePooling2D,Dropout
from keras.src.optimizers import Adam
import joblib
import pathlib
import numpy as np





def fit(X_tr,y_tr,X_val,y_val):
    base_model = resnet.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(6, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    model.summary()
    history = model.fit(
    X_tr,y_tr,
    validation_data=(X_val,y_val),
    epochs=20,
    batch_size=32
        )
    return history,model

def save_model(model,model_save_path):
    joblib.dump(model, model_save_path+'/model.joblib')


def main():
    curr_dir=pathlib.Path().cwd().as_posix()
    data_path=curr_dir+'/data/processed/'
    X_tr=np.load(data_path+'X_tr.npy')
    y_tr=np.load(data_path+'y_tr.npy')
    X_val=np.load(data_path+'X_val.npy')
    y_val=np.load(data_path+'y_val.npy')
    model_output_path=curr_dir+'/models'
    
   
    history,model=fit(X_tr,y_tr,X_val,y_val)
    save_model(model,model_output_path)



if __name__ =='__main__':
    main()