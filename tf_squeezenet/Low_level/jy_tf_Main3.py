# -*-coidng:utf-8-*-
"""
    output: accuracy/probability each class
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from Low_level import jy_tf_function as myFunc  # tf 관련 함수 라이브러리
from Low_level import jy_tf_init as init
from Low_level import jy_tf_model
from Low_level import jy_tf_nnLib as nnLib
import numpy as np
import tensorflow as tf


def main():
    start = time.time()

    # TASK='train'
    # placeholder(dtype, shape=None, name=None)
    image_batch = myFunc.read_images_from_list(init.trainTxt)  # fileTxt: image file list txt
    test_batch = myFunc.read_images_from_list(init.testTxt)

    test_valid = myFunc.load_test_image2(init.testTxt)
    # pbTest = myFunc.load_test_image('E:/1.pear/TestSet_1/pb/', 0)

    x = tf.placeholder(tf.float32, [None, init.picSize, init.picSize, init.channel], name='X')
    y_ = tf.placeholder(tf.float32, [None, init.num_classes], name='Y')
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # 실행할 함수 이름 설정
    model = getattr(jy_tf_model, init._model_)
    py_x = model(x, phase_train)

    output = nnLib.training(py_x, y_)
    predict_op = tf.argmax( py_x, 1 )
    prob = tf.nn.softmax( py_x )

    saver = tf.train.Saver()

    # epoch = mDefault.epoch
    iteration = init.iteration
    f = open(init.save_accuracy, 'w')
    ft = open(init.save_valid, 'w')

    with tf.Session() as sess:
        tf.global_variables_initializer().run(feed_dict={phase_train: True})
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        test_tensor = sess.run(test_batch)

        epoch = 0
        for i in range(iteration):
            img_tensor = sess.run(image_batch)
            sess.run( output, feed_dict = {x: img_tensor[0], y_: img_tensor[1],
                                                                phase_train: True} )
            # validation
            term = int(init.total_image / init.batch_size)
            if i % term == 0:
                result, predict_prob = sess.run( [predict_op, prob],feed_dict = {x: test_tensor[0], phase_train: False} )
                validation = np.mean(np.argmax(test_tensor[1], axis=1) == result)
                print(epoch, '[  accuracy : %.4f' % validation, '] ', 'prob: [', *predict_prob[init.batch_size-1], ']')
                epoch += 1

                # print("===================================================================================")
        f.close()
        ft.close()

        # print the running time
        duration = time.time() - start
        hour = int(duration) / 3600
        minute = int(duration % 3600) / 60
        second = (duration % 3600) % 60

        # test results, not validation
        confusion_matrix = [[0 for _ in range(init.num_classes)] for _ in range(init.num_classes)]
        precision = 0
        iteration = int(init.num_classes * init.test_cnt / init.test_batchSize)
        for t in range(iteration):
            print(t)
            valid_tensor = sess.run(test_valid)
            pred = sess.run(predict_op, feed_dict={x: valid_tensor[0], phase_train: False})
            answer = np.argmax(valid_tensor[1], axis=1)

            for c in range(len(pred)):
                confusion_matrix[int(pred[c])][int(answer[c])] += 1
                if (int(pred[c]) == int(answer[c])):
                    precision += 1

        for row in confusion_matrix:
            print(row)
        print('accuracy: %.4f' % (precision / init.total_test_cnt))
        myFunc.save_confusion_matrix(confusion_matrix)

        path = init.save_saver + init.saver_name
        saver.save(sess, path)

        print(int(hour), 'h ', int(minute), 'm ', second, 's')

        coord.request_stop()
        coord.join(threads)

        print(init.saver_name)
        print(init._model_)



