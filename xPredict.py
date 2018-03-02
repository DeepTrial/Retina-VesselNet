import numpy as np
import configparser
from matplotlib import pyplot as plt
import cv2
import glob
from keras.models import model_from_json
from keras.models import Model
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys
sys.path.insert(0, './lib/')
# help_functions.py
from lib.help_functions import *
# extract_patches.py
from lib.extract_patches import recompone_overlap
from lib.extract_patches import get_data_predict_overlap
from lib.pre_processing import *




#========= CONFIG FILE TO READ FROM =======
config = configparser.RawConfigParser()
config.read('configuration.txt')
#===========================================s
#run the training on invariant or local
path_data = config.get('data paths', 'path_local')

#original test images (for FOV selection)
DRIVE_test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')


#the border masks provided by the DRIVE
#DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
#test_border_masks = load_hdf5(DRIVE_test_border_masks)
# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
#the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)
#model name
name_experiment = config.get('experiment name', 'name')
path_experiment = './DataSet/'
#N full images to be predicted
Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
#Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
#====== average mode ===========
average_mode = config.getboolean('testing settings', 'average_mode')
num_lesion_class = int(config.get('data attributes','num_lesion_class'))
Nimgs = int(config.get('data attributes','total_data_test'))

imgList = glob.glob('./TestFold/origin/'+'*.'+config.get('predict settings','img_type'))

Nimgs_train=len(imgList)

count=1
for i in imgList:
    test_imgs_original = plt.imread(i)
    #test_imgs_original=cv2.imread(i)
    height, width = test_imgs_original.shape[:2]
    test_imgs_original=test_imgs_original[:,:,1]*0.75+test_imgs_original[:,:,0]*0.25
    test_imgs_original=np.reshape(test_imgs_original,(height,width,1))
    full_img_height = height#int(height*0.5)
    full_img_width = width#int(width*0.5)

    name=i.split('\\')[-1]
    picname=name.split('.')[0]

    print(picname,'start')
#============ Load the data and divide in patches
    patches_imgs_test = None
    new_height = None
    new_width = None

    patches_imgs_test, new_height, new_width,original_adjust = get_data_predict_overlap(
        imgPredict = test_imgs_original,  #original
        patch_height = patch_height,
        patch_width = patch_width,
        stride_height = stride_height,
        stride_width = stride_width,
        num_lesion = num_lesion_class,
        total_data = Nimgs
        )



#================ Run the prediction of the patches ==================================
    best_last = config.get('testing settings', 'best_last')
#Load the saved model
    model = model_from_json(open(path_experiment+name_experiment +'_architecture.json').read())
    model.load_weights(path_experiment+name_experiment + '_'+best_last+'_weights.h5')
#Calculate the predictions
    predictions = model.predict(patches_imgs_test, batch_size=32, verbose=1)
    print("\npredicted images size :")
    print(predictions.shape)
    print(np.max(predictions[:,:,0]))
    print('test',np.max(predictions),np.min(predictions))
#===== Convert the prediction arrays in corresponding images
    pred_patches = pred_to_imgs(predictions,"original")
    pred_patches=pred_patches.transpose(0,3,1,2)
    print(pred_patches.shape)


#========== Elaborate and visualize the predicted images ====================
    pred_imgs = None
    orig_imgs = None
   
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width,patch_height,patch_width, stride_height, stride_width)# predictions
    #orig_imgs = my_PreProc(test_imgs_original)    #originals

    result=0
    #orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
    pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]


    visualize_all(group_images(pred_imgs[:, 0:1, :, :], N_visual), './TestFold/result/' + picname + "_manul1.jpg")

    count=count+1
    original_adjust = original_adjust.transpose(0, 2, 3, 1)[0]
    all=np.zeros((height*2, width))
    original_adjust=np.reshape( original_adjust,( original_adjust.shape[0], original_adjust.shape[1]))
    all[:height,:,]=(original_adjust*255).astype(np.uint8)

    gtname=name[:3]
    pred_gt=pred_imgs.transpose(0,2,3,1)[0]

    _,pred_gt=cv2.threshold( pred_gt[:,:,0],0.3,1,cv2.THRESH_BINARY)
    all[height:2 * height, :]=(pred_gt*255).astype(np.uint8)
    #all[height:2 * height, :, 1]=(pred_gt[:,:,1]*255)#.astype('uint8')
    imgTotal = Image.fromarray(np.uint8(all))
    imgTotal.save('./TestFold/result/'+name+'.jpg')
    print(i,result)
