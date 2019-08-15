import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os

Flags=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("train",0,"指定是否训练")
#读取数据集的数字
def get_mnist_data(size):
    minist=input_data.read_data_sets("./data/mnist/input_data/",one_hot=True)
    x,y=minist.train.next_batch(size)

    # 显示图片
    #fig,ax=plt.subplots(nrows=4,ncols=5,sharex="all",sharey="all")
    # ax = ax.flatten()
    # for i in range(20):
    #     image=x[i].reshape(28,28)
    #     ax[i].imshow(image,cmap='Greys')
    # plt.tight_layout()
    # plt.show()
    return [x,y]

#读取自己的图片
def get_img_data(path):

    x=np.zeros((10,32*32))
    y=np.zeros((10,10))
    i=0
    for filename in os.listdir(path):
        img = cv2.imread(path + "/" + filename)
        img=cv2.resize(img,(32,32))
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY);
        x[i]=img.reshape(1,32*32)
        x[i]=x[i]/255.0
        index=filename.index('.')
        number=int(filename[index-1])
        y[i][number]=1
        i+=1
    return x,y
#初始权重
def get_weight(shape):
    return tf.Variable(tf.random_normal(shape,mean=0,stddev=0.1))


#处理图片28*28->32*32
def do_image(x,batch):
    xnew=np.zeros((batch,32*32))
    for i in range(batch):
        image=x[i].reshape(28,28)
        newimage=cv2.resize(image, (32, 32))
        xnew[i]=newimage.reshape(1,32*32)
    return xnew
def model():
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 1024])
        y_true = tf.placeholder(tf.int32, [None, 10])

    #卷积层 f=5*5 s=1 c=6
    with tf.variable_scope("conv1"):
        conv1_w=get_weight([5,5,1,6])
        conv1_b=get_weight([6])
        #装换为[None,28,28,1]的输入
        x_reshape=tf.reshape(x,[-1,32,32,1])
        #32*32*1->28*28*6
        conv1_re=tf.nn.relu(tf.nn.conv2d(x_reshape,conv1_w,strides=[1,1,1,1],padding="VALID")+conv1_b)
        # 28*28*6->14*14*6
        maxpool1_re = tf.nn.max_pool(conv1_re,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    with tf.variable_scope("conv2"):
        conv2_w = get_weight([5, 5, 6, 16])
        conv2_b = get_weight([16])
        # 14*14*6->10*10*16
        conv2_re = tf.nn.relu(tf.nn.conv2d(maxpool1_re, conv2_w, strides=[1, 1, 1, 1], padding="VALID") + conv2_b)
        # 10*10*16->5*5*16
        maxpool2_re = tf.nn.max_pool(conv2_re, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
   # 全连接层[None, 5, 5, 16] - -->[None, 5 * 5 * 16] * [5 * 5* 16, 120] + [10] = [None, 120]
    with tf.variable_scope("FC1"):
        fc_w1 = get_weight([5*5*16,120])
        fc_b1 = get_weight([120])
        fcx_reshape1=tf.reshape(maxpool2_re,[-1,5*5*16])
        y_predict1 = tf.nn.relu(tf.matmul(fcx_reshape1, fc_w1) + fc_b1)
    # [-1,120]*[120*84]
    with tf.variable_scope("FC2"):
        fc_w2 = get_weight([120, 84])
        fc_b2 = get_weight([84])
        y_predict2 = tf.nn.relu(tf.matmul(y_predict1, fc_w2) + fc_b2)
    #[-1,84]*[84*10]
    with tf.variable_scope("FC3"):
        fc_w3 = get_weight([84, 10])
        fc_b3= get_weight([10])
        y_predict = tf.matmul(y_predict2, fc_w3) + fc_b3
    return x,y_true,y_predict

#识别数字
def recongize_number():
    mnist=input_data.read_data_sets("./data/mnist/input_data/",one_hot=True)
    x, y_true, y_predict = model()
    #定义损失
    with tf.variable_scope("loss"):
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))
    #利用梯度下降
    with tf.variable_scope("gradient"):
        train_op=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    # 计算准确率
    with tf.variable_scope("acc"):
        equal_list=tf.equal(tf.argmax(y_true,1),tf.argmax(y_predict,1))
        accurancy=tf.reduce_mean(tf.cast(equal_list,tf.float32))
    #tensorborad
    tf.summary.scalar("loss",loss)
    tf.summary.scalar("acc",accurancy)

    merged = tf.summary.merge_all()

    mysave=tf.train.Saver()
    #初始化
    init_op=tf.global_variables_initializer()
    # 开启回话运行
    with tf.Session() as sess:
        sess.run(init_op)
        # 训练
        filewriter = tf.summary.FileWriter("./log", graph=sess.graph)
        if Flags.train==1:
            ff = os.path.exists("./mysaver/checkpoint")
            print(ff)
            if ff:
                mysave.restore(sess, "./mysaver/my_model")
            #训练mnist的数据
            for i in range(8000):
                mnist_x, mnist_y = mnist.train.next_batch(100)
                #转换为由28*28->32*32
                mnist_x=do_image(mnist_x,100)
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
                summary=sess.run(merged,feed_dict={x: mnist_x, y_true: mnist_y})
                filewriter.add_summary(summary,i)
                # 输出精度
                print("第%d次，准确度为：%f" % (i + 1, sess.run(accurancy, feed_dict={x: mnist_x, y_true: mnist_y})))
                if (i+1)%500==0:
                    mysave.save(sess,"./mysaver/my_model")
        elif Flags.train==2:
            #使用mnist训练的参数继续训练
            ff=os.path.exists("./mysaver/checkpoint")
            if ff:
                mysave.restore(sess, "./mysaver/my_model")

            #训练自己写的图片
            for i in range(10):
                mydata_x,mydata_y=get_img_data("./data/mydata/train/")
                sess.run(train_op, feed_dict={x: mydata_x, y_true: mydata_y})
                summary = sess.run(merged, feed_dict={x: mydata_x, y_true: mydata_y})
                filewriter.add_summary(summary, i)
                # 输出精度
                print("第%d次，准确度为：%f" % (i + 1, sess.run(accurancy, feed_dict={x: mydata_x, y_true: mydata_y})))
            mysave.save(sess, "./mysaver/my_model")
        elif Flags.train == 0:
            #测试mnist数据
            mysave.restore(sess, "./mysaver/my_model")
            for i in range(100):
                x_test, y_test = mnist.train.next_batch(1)
                # 转换为由28*28->32*32
                x_test = do_image(x_test, 1)
                print("第%d张图片，手写数字图片目标是:%d, 预测结果是:%d" % (
                    i+1,
                    tf.argmax(y_test, 1).eval(),
                    tf.argmax(sess.run(y_predict, feed_dict={x: x_test, y_true: y_test}), 1).eval()
                ))
        else:
            # 测试自己的图片
            mysave.restore(sess, "./mysaver/my_model")
            mydata_x, mydata_y = get_img_data("./data/mydata/test/")
            for i in range(10):

                print("第%d张图片，手写数字图片目标是:%d, 预测结果是:%d" % (
                    i + 1,
                    tf.argmax(mydata_y[i].reshape(1,10), 1).eval(),
                    tf.argmax(sess.run(y_predict, feed_dict={x: mydata_x[i].reshape(1,32*32), y_true: mydata_y[i].reshape(1,10)}), 1).eval()
                ))
                plt.imshow(mydata_x[i].reshape(32, 32), cmap='Greys')
                plt.title(str(tf.argmax(sess.run(y_predict, feed_dict={x: mydata_x[i].reshape(1, 32 * 32),
                                                                       y_true: mydata_y[i].reshape(1, 10)}),
                                        1).eval()))  # 图像标题
                plt.show()
if __name__ == "__main__":
    recongize_number()

