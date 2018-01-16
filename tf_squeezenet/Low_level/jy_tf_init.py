import os

from Low_level import jy_makeList as list
from Low_level import jy_tf_Main
from Low_level import restore

phase_train = True # training / test
listOnOff = False # choose to make dataset dir txt file

root = 'D:/myFolder/'
folderList = ['1', '2', '3', '4'] # real name of folders name in root

trainTxt = root+'imgTrainingList.txt' # txt file of training image directory
testTxt = root+'imgTestList.txt'# txt file of test image directory

tr_cnt=1000 # of images for training each class
test_cnt =200 # of images for test each class
saver_name='datasetName' #input the dataset name

##################################
#network setting
_model_ = 'SqueezeNet' #model name
filter = 224  #initial # of filteres
picSize = 224  # size of image
imgType= 'jpg' #'png' #type of image

batch_size = 32
test_batchSize = 10
epoch = 256


#####################################
############auto setting#############

channel = 3
num_classes = len(folderList)

total_image = num_classes*tr_cnt
iteration = int(epoch * total_image/batch_size)
total_test_cnt = num_classes * test_cnt

#save log
save_accuracy = root + 'accuracy.txt'
save_valid = root + 'validation.txt'
save_saver = root + '0.ckpt/' +_model_+'/'
save_confusion = root + 'confusion_matrix.txt'
save_fp = root + 'fp.txt'


if __name__ == '__main__':
    if listOnOff:
        list.makelist_main()
    if not os.path.exists(save_saver):
        os.mkdir(save_saver)

    if(phase_train):
        jy_tf_Main.main()
    else:
        restore.test2()

