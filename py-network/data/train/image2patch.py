import matplotlib.pyplot as plt
import numpy as np
from data.preprocess import preprocess
import cv2

def check_coord(x,y,h,w,patch_size):
    if x-patch_size/2>0 and x+patch_size/2<h and y-patch_size/2>0 and y+patch_size/2<w:
        return True
    return False

def image2patch(image_path,patch_num,patch_size,patch_threshold,train_patch_dir,train_groundtruth_dir,train_mask_dir,training=True,show=True):
    image_name=image_path.split("/")[-1].split("_")[0]

    image=plt.imread(image_path)

    groundtruth=plt.imread(train_groundtruth_dir+image_name+"_manual1.gif")
    groundtruth=np.where(groundtruth>0,1,0)

    mask=plt.imread(train_mask_dir+image_name+"_training_mask.gif")
    mask=np.where(mask>0,1,0)
  
    image=preprocess(image,mask)
    #image_binary=0.8*image[:,:,1]+0.2*image[:,:,2]

    image_show=image.copy()
    groundtruth_show=np.zeros_like(image)
    groundtruth_show[:,:,0]=groundtruth.copy()
    groundtruth_show[:,:,1]=groundtruth.copy()
    groundtruth_show[:,:,2]=groundtruth.copy()

    sample_count=0
    sample_index=0
  
    sample_point=np.where(groundtruth==1)     # generate sample point (生成采样中心点)

    state = np.random.get_state()      # shuffle the coord (打乱顺序，模拟随机采样)
    np.random.shuffle(sample_point[0])
    np.random.set_state(state)
    np.random.shuffle(sample_point[1])

    patch_image_list=[]
    patch_groundtruth_list=[]
    plt.ion()
    while sample_count<patch_num and sample_index<len(sample_point[0]):
        x,y=sample_point[0][sample_index],sample_point[1][sample_index]
        if check_coord(x,y,image.shape[0],image.shape[1],patch_size):
            if np.sum(mask[x-patch_size//2:x+patch_size//2,y-patch_size//2:y+patch_size//2])>patch_threshold:     #select according to the threshold
       
                patch_image_binary=image[x-patch_size//2:x+patch_size//2,y-patch_size//2:y+patch_size//2,:]   # patch image
                patch_groundtruth=groundtruth[x-patch_size//2:x+patch_size//2,y-patch_size//2:y+patch_size//2]       # patch mask
                #patch_image_binary=np.asarray(0.25*patch_image[:,:,2]+0.75*patch_image[:,:,1])         # B*0.25+G*0.75, which enhance the vessel (增强血管的对比度)
                patch_groundtruth=np.where(patch_groundtruth>0,255,0)
    
                #patch_image_binary =cv2.equalizeHist((patch_image_binary*255.0).astype(np.uint8))/255.0
        
                patch_image_list.append(patch_image_binary)    # patch image
                patch_groundtruth_list.append(patch_groundtruth)             # patch mask
            if show:
                cv2.rectangle(image_show, (y-patch_size//2,x-patch_size//2,), (y+patch_size//2,x+patch_size//2), (0,1,0), 2)  #draw the illustration
                cv2.rectangle(groundtruth_show, (y-patch_size//2,x-patch_size//2,), (y+patch_size//2,x+patch_size//2), (0,1,0), 2)
            sample_count+=1
    
        if show:                                 # visualize the sample process(可视化采样过程，会很慢！)
            plt.figure(figsize=(15,15))
            plt.title("processing: %s"%image_name)
            plt.subplot(121)
            plt.imshow(image_show,cmap=plt.cm.gray)   # processd image
            plt.subplot(122)
            plt.imshow(groundtruth_show,cmap=plt.cm.gray)  #groundtruth of the image, patch is showed as the green square (绿色的方框表示采样的图像块)
            plt.show()
        sample_index+=1
    plt.ioff()
    for i in range(len(patch_image_list)):
        if training==True:
            plt.imsave(train_patch_dir+image_name+"-"+str(i)+"-img.jpg",patch_image_list[i])
            #print(patch_mask_list[i])
            plt.imsave(train_patch_dir+image_name+"-"+str(i)+"-groundtruth.jpg",(patch_groundtruth_list[i]/225.0).astype(np.uint8),cmap = plt.cm.gray)
        else:
            plt.imsave(train_patch_dir+image_name+"_"+str(i)+"_val_img.jpg",patch_image_list[i])
            #print(patch_mask_list[i])
            plt.imsave(train_patch_dir+image_name+"_"+str(i)+"_val_groundtruth.jpg",(patch_groundtruth_list[i]/225.0).astype(np.uint8),cmap = plt.cm.gray)
