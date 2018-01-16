#! /usr/bin/env_python
# -*-coding:utf-8-*-
"""
    make list for training or test from directory
    output: ...list.txt

"""
import glob
import random
from Low_level import jy_tf_init as init


def make_list_glob(path, label):
    roots = init.root
    # training
    cnt = 0
    if(init.treeOnOff):
        files = glob.glob(roots + path + '/aug/*')
    else:
        files = glob.glob(roots + path + '/*')
    trainingList = []

    for file in files:
        if cnt < init.tr_cnt:
            name = '%s\t' % file + '%d\n' % label
            trainingList.append(name)
            cnt += 1

    # test
    files.sort(reverse=True)
    testList = []
    cnt = 0
    for file in files:
        if cnt < init.test_cnt:
            name = '%s\t' % file + '%d\n' % label
            testList.append(name)
            cnt += 1

    return trainingList, testList


def makelist_main():

    imgTrainingList = []
    imgTestList = []

    for c in range(init.num_classes):
        list1, list2 = make_list_glob(init.folderList[c], c)
        imgTrainingList += list1
        imgTestList += list2

    random.shuffle(imgTrainingList)
    file_name = init.trainTxt
    f = open(file_name, 'w')

    # write list on notepad
    for line in imgTrainingList:
        f.write(line)
    f.close()
    print('training')

    random.shuffle(imgTestList)
    file_name = init.testTxt
    f = open(file_name, 'w')
    for line in imgTestList:
        f.write(line)
    f.close()
    print('test')

def makelist_main2():
    """
    train/test 폴더가 분리되어 있을 경우
    한번에 리스트 만들고 txt파일 저장하는 것까지 함
    """

    imgTrainingList = []
    imgTestList = []
    cnt = 0

    for c in range(init.num_classes):
        roots = init.root
        # training

        trainingList = []
        files = glob.glob(roots + init.folderList[c] + '/train/*')
        for file in files:
            if cnt<init.tr_cnt:
                name = '%s\t' % file + '%d\n' % c
                trainingList.append(name)
            cnt+=1

        # test
        testList = []
        files = glob.glob(roots + init.folderList[c] + '/test/*')
        cnt=0

        for file in files:
            if cnt<init.test_cnt:
                name = '%s\t' % file + '%d\n' % c
                testList.append(name)
            cnt+= 1
        imgTrainingList += trainingList
        imgTestList += testList

    random.shuffle(imgTrainingList)
    file_name = init.trainTxt
    f = open(file_name, 'w')

    # write list on notepad
    for line in imgTrainingList:
        f.write(line)
    f.close()
    print('training')

    file_name = init.testTxt
    f = open(file_name, 'w')
    for line in imgTestList:
        f.write(line)
    f.close()
    print('test')