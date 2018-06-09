
import vesselModel
import sys,random
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils import plot_model
from lib.help_functions import *
from keras.models import model_from_json
from lib.pre_processing import *
import matplotlib.pyplot as plt
import configparser
import numpy as np
from keras.callbacks import ReduceLROnPlateau
#function to obtain data for training/testing (validation)
K.set_image_dim_ordering('th')
NewTrain=True
ChangeReg=False
config = configparser.RawConfigParser()
config.read('./configuration.txt')
Tensorflow=int(config.get('data attributes', 'Tensorflow'))


def CenterSampler(img_h,img_w,patch_h,patch_w,class_weight,mlist,Nimgs):
    class_weight=class_weight/np.sum(class_weight)
    p = random.uniform(0,1)
    psum=0
    label=0
    for i in range(class_weight.shape[0]):
        psum=psum+class_weight[i]
        if p<psum:
            label=i
            break
    # if label==class_weight.shape[0]-1:
    #     i_center = random.randint(0,Nimgs-1)
    #     x_center = random.randint(0+int(patch_w/2),img_w-int(patch_w/2))
    #      # print "x_center " +str(x_center)
    #     y_center = random.randint(0+int(patch_h/2),img_h-int(patch_h/2))
    # else:
    t=mlist[label]
    cid=random.randint(0,t[0].shape[0]-1)
    i_center=t[0][cid]
    y_center=t[1][cid]+random.randint(0-int(patch_w/2),0+int(patch_w/2))
    x_center=t[2][cid]+random.randint(0-int(patch_w/2),0+int(patch_w/2))
        #mask_shape=train_masks.shape[3]

    if y_center<patch_w/2:
        y_center=patch_w/2
    elif y_center>img_h-patch_w/2:
        y_center=img_h-patch_w/2

    if x_center<patch_w/2:
        x_center=patch_w/2
    elif x_center>img_w-patch_w/2:
        x_center=img_w-patch_w/2

    return i_center,x_center,y_center


def Active_Generate(train_imgs,train_masks,patch_h,patch_w,batch_size,N_subimgs,N_imgs,class_weight,mlist):
    while 1:
        img_h=train_imgs.shape[2]
        img_w=train_imgs.shape[3]
        for t in range(int(N_subimgs*N_imgs/batch_size)):
            X=np.zeros([batch_size,1,patch_h,patch_w])
            Y=np.zeros([batch_size,patch_h*patch_w,num_lesion_class+1])
            for j in range(batch_size):
                [i_center,x_center,y_center]=CenterSampler(img_h,img_w,patch_h,patch_w,class_weight,mlist,N_imgs)
                patch = train_imgs[i_center,:,int(y_center-patch_h/2):int(y_center+patch_h/2),int(x_center-patch_w/2):int(x_center+patch_w/2)]
                patch_mask = train_masks[i_center,:,int(y_center-patch_h/2):int(y_center+patch_h/2),int(x_center-patch_w/2):int(x_center+patch_w/2)]
                X[j,:,:,:]=patch
                Y[j,:,:]=masks_Unet(np.reshape(patch_mask,[1,num_lesion_class,patch_h,patch_w]),num_lesion_class)
            # if t%10==0:
            #     sys.stdout.write(' '*4+'\r'+str(t).zfill(4))
            #     sys.stdout.flush()
            yield (X, Y)

def SampleTest(gen,batch_size,patch_h,patch_w):
    (X,Y)=next(gen)
    visualize_all(group_images(X[:, 0:1, :, :], 4), './DataSet/patchSample.jpg')
    Y=np.reshape(Y,[batch_size,patch_h,patch_w,num_lesion_class+1])
    Y=Y.transpose(0,3,1,2)
    visualize_all(group_images(Y[:,0:1,:,:],4), './DataSet/groundtruthSample.jpg')
    # for i in range(batch_size):
    #     temp=X.transpose(0,2,3,1)[i]
    #     plt.imshow(temp[:,:,0])
    #     #pl.imshow(X[i])
    #     plt.show()
    #     plt.imshow(np.reshape(Y,[batch_size,patch_h,patch_w,num_lesion_class+1])[i,:,:,0],cmap='gray')
    #     plt.show()


patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
num_lesion_class = int(config.get('data attributes', 'num_lesion_class'))
patch_h=patch_height
patch_w=patch_width
N_subimgs = int(config.get('training settings', 'N_subimgs'))
inside_FOV = config.getboolean('training settings', 'inside_FOV')
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
name_experiment = config.get('experiment name', 'name')
#filename='./DRIVE/training/train'
best_last = config.get('testing settings', 'best_last')
path_experiment = './DataSet/'
N_imgs=int(config.get('training settings', 'full_images_to_train'))

path_data = config.get('data paths', 'path_local')
train_imgs_original = path_data + config.get('data paths', 'train_imgs_original')
train_groundTruth = path_data + config.get('data paths', 'train_groundTruth')
train_imgs_original = load_hdf5(train_imgs_original)#[img_id:img_id+1]
train_masks=np.zeros([N_imgs,num_lesion_class,train_imgs_original.shape[2],train_imgs_original.shape[3]])
train_masks = load_hdf5(train_groundTruth )

#mlist=[np.where(train_masks[:,0,:,:]==np.max(train_masks[:,0,:,:]))]



Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_groundTruth = path_data + config.get('data paths', 'test_groundTruth')
test_imgs_original = load_hdf5(test_imgs_original)
test_masks=np.zeros([Imgs_to_test,num_lesion_class,train_imgs_original.shape[2],train_imgs_original.shape[3]])
test_masks = load_hdf5(test_groundTruth )#masks always the same



train_imgs = my_PreProc(train_imgs_original)
#train_imgs = train_imgs_original
train_masks = train_masks/255.
#train_imgs = train_imgs[:,:,0:501,:]  #cut bottom and top so now it is 565*565
#train_masks = train_masks[:,:,0:501,:]  #cut bottom and top so now it is 565*565

test_imgs =my_PreProc(test_imgs_original)
#test_imgs = test_imgs_original
test_masks = test_masks/255.
#test_imgs = test_imgs[:,:,0:501,:]  #cut bottom and top so now it is 565*565
#test_masks = test_masks[:,:,0:501,:]  #cut bottom and top so now it is 565*565

#gen=generate_arrays_from_file(train_imgs,train_masks,patch_height,patch_width,batch_size,N_subimgs,N_imgs)
class_weight=np.array([1.0])#,30.0,90.0,50.0])#,60.0],20.0,90.0])#[10.0,30.0,20.0,60.0])#[10.0,30.0,20.0,60.0]
class_weight=class_weight/np.sum(class_weight)
test_class_weight=np.array([1.0])#,1.0,1.0,1.0])#,0.0,0.0,1.0])


mlist=[np.where(train_masks[:,0,:,:]==np.max(train_masks[:,0,:,:]))]#,
#print("mlist shape",mlist[0].shape)
       # np.where(train_masks[:,1,:,:]==np.max(train_masks[:,1,:,:])),
       # np.where(train_masks[:,2,:,:]==np.max(train_masks[:,2,:,:])),
       # np.where(train_masks[:,3,:,:]==np.max(train_masks[:,3,:,:]))]
# if np.max(train_masks[:,0,:,:])>1.0:
#     for i in range(len(mlist[0][0])):
#         train_masks[mlist[0][0][i],0,mlist[0][1][i],mlist[0][2][i]]=1.0
#     mlist=[np.where(train_masks[:,0,:,:]==np.max(train_masks[:,0,:,:]))]

gen=Active_Generate(train_imgs,train_masks,patch_height,patch_width,batch_size,N_subimgs,N_imgs,class_weight,mlist)
SampleTest(gen,batch_size,patch_height,patch_width)

#one time test_sampling
test_mlist=[np.where(test_masks[:,0,:,:]==np.max(test_masks[:,0,:,:]))]#,
            # np.where(test_masks[:,1,:,:]==np.max(test_masks[:,1,:,:])),
            # np.where(test_masks[:,2,:,:]==np.max(test_masks[:,2,:,:])),
            # np.where(test_masks[:,3,:,:]==np.max(test_masks[:,3,:,:]))]
# if np.max(test_masks[:,0,:,:])>1.0:
#     for i in range(len(mlist[0][0])):
#         test_masks[test_mlist[0][0][i],0,test_mlist[0][1][i],test_mlist[0][2][i]]=1.0
#     test_mlist=[np.where(test_masks[:,0,:,:]==np.max(test_masks[:,0,:,:]))]

test_gen=Active_Generate(test_imgs,test_masks,patch_height,patch_width,batch_size,N_subimgs,Imgs_to_test,test_class_weight,test_mlist)



if NewTrain:
    model = vesselModel.R_Unet(1, patch_height, patch_width)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=[vesselModel.fbeta_score,vesselModel.dice_coef])#['categorical_accuracy'])
    #plot_model(model, to_file='/home/huangyj/Documents/retina-unet-Binary/model.png',show_shapes=True)   #check how the model looks like
    plot_model(model, to_file='model.png',show_shapes=True)
    json_string = model.to_json()
    open('./DataSet/'+name_experiment +'_architecture.json', 'w').write(json_string)
elif ChangeReg:
    model = vesselModel.get_unet(1, patch_height, patch_width)
    model.load_weights('./DataSet/'+name_experiment + '_'+best_last+'_weights.h5')
    model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=[vesselModel.dice_coef])
    #plot(model, to_file='./'+name_experiment+'/'+name_experiment + '_model.png')   #check how the model looks like
    json_string = model.to_json()
    open('./DataSet/'+name_experiment +'_architecture.json', 'w').write(json_string)
else:
    model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
    model.load_weights('./DataSet/'+name_experiment + '_'+best_last+'_weights.h5')
    model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=[vesselModel.dice_coef])#'categorical_accuracy'])


checkpointer = ModelCheckpoint(filepath='./DataSet/'+name_experiment +'_best_weights.h5',
                               verbose=1,
                               monitor='val_loss',
                               mode='auto',
                               save_best_only=True)


val_num=76800

history = vesselModel.LossHistory()
lrmod=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
hist=model.fit_generator(gen,#generate_arrays_from_file(train_imgs,train_masks,patch_height,patch_width,batch_size,N_subimgs,N_imgs),
                    epochs=N_epochs,
                    steps_per_epoch=N_subimgs*N_imgs/batch_size,
                    verbose=1,
                    callbacks=[checkpointer,history,lrmod],
                    validation_data=test_gen,
                    validation_steps=int(N_subimgs*Imgs_to_test/batch_size))

history.loss_plot('epoch')
with open('./DataSet/loss_plot.txt','w') as f:
    f.write(str(hist.history))

model.save_weights('./DataSet/'+name_experiment +'_last_weights.h5', overwrite=True)

