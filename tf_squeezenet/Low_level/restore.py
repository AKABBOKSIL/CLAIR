from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Low_level import jy_tf_function as myFunc
from Low_level import jy_tf_init as init
from Low_level import jy_tf_model
import numpy as np
import tensorflow as tf


def test2():
    test_valid = myFunc.load_test_image2(init.testTxt)

    x = tf.placeholder(tf.float32, [None, init.picSize, init.picSize, init.channel], name='X')

    phase_train = tf.placeholder(tf.bool, name='phase_train')
    model = getattr(jy_tf_model, init._model_)
    py_x = model(x, phase_train)

    #output = nnLib.training(py_x, y_)
    predict_op = tf.argmax(py_x, 1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run(feed_dict={phase_train: True})
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Restore model weights from previously saved model
        print(init.save_saver+ init.saver_name)
        saver.restore(sess, init.save_saver+ init.saver_name)
        print('restore')

        confusion_matrix = [[0 for _ in range(init.num_classes)] for _ in range(init.num_classes)]
        precision = 0
        iteration = int(init.num_classes * init.test_cnt / init.test_batchSize)
        for t in range(iteration):
            print(t)
            valid_tensor = sess.run(test_valid)
            pred = sess.run(predict_op, feed_dict={x: valid_tensor[0], phase_train: False})
            list = np.argmax(valid_tensor[1], axis=1)

            for c in range(len(pred)):
                confusion_matrix[int(pred[c])][int(list[c])] += 1
                if (int(pred[c]) == int(list[c])):
                    precision += 1

        for row in confusion_matrix:
            print(row)
        myFunc.save_confusion_matrix(confusion_matrix)
        print('accuracy: %.4f' % (precision / init.total_test_cnt))

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    test2()