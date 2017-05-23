import tensorflow as tf
from gen_train import get_train_text_and_image
import numpy as np
from gen_test import get_test_text_and_image

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256

MAX_CAPTCHA = 16
CHAR_SET_LEN = 95


X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32) # 防止过拟合

"""
定义卷积神经网络
"""
def create_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    """
    定义卷积层
    """
    """第一层"""
    """
    定义卷积核
    """
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    """偏置值"""
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.conv2d(x, w_c1, [1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.bias_add(conv1, bias=b_c1)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    # [32,128,32]

    """第二层"""
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 128]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([128]))
    conv2 = tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b_c2)
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    # [16, 64, 128]

    """第三层"""
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 256]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([256]))
    conv3 = tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b_c3)
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    # [8, 32, 256]

    """第四层"""
    w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 256, 512]))
    b_c4 = tf.Variable(b_alpha * tf.random_normal([512]))
    conv4 = tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.bias_add(conv4, b_c4)
    conv4 = tf.nn.relu(conv4)
    conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv4 = tf.nn.dropout(conv4, keep_prob)
    # [4, 16, 512]


    """第五层"""
    w_c5 = tf.Variable(w_alpha * tf.random_normal([3, 3, 512, 1024]))
    b_c5 = tf.Variable(b_alpha * tf.random_normal([1024]))
    conv5 = tf.nn.conv2d(conv4, w_c5, strides=[1, 1, 1, 1], padding='SAME')
    conv5 = tf.nn.bias_add(conv5, b_c5)
    conv5 = tf.nn.relu(conv5)
    conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv5 = tf.nn.dropout(conv5, keep_prob)
    # [2, 8, 1024]

    """
    全连接层
    """
    """第六层"""
    w_d = tf.Variable(w_alpha * tf.random_normal([16384, 4096]))
    b_d = tf.Variable(b_alpha * tf.random_normal([4096]))
    dense = tf.reshape(conv5, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    """第七层"""
    w_out = tf.Variable(w_alpha * tf.random_normal([4096, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out


def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # int gray = (int) (0.3 * r + 0.59 * g + 0.11 * b);
        return gray
    else:
        return img

# 文本转向量
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('长度超限')
    while len(text)< MAX_CAPTCHA:
        text = text + " "

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)

    def char2pos(c):
        try:
            k = ord(c)-ord(' ')
            if k > 95:
                k = ord(' ')
        except:
            raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        try:
            vector[idx] = 1
        except:
            pass
    return vector



# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        # if char_idx < 10:
        #     char_code = char_idx + ord('0')
        # elif char_idx < 36:
        #     char_code = char_idx - 10 + ord('A')
        # elif char_idx < 62:
        #     char_code = char_idx - 36 + ord('a')
        # elif char_idx == 62:
        #     char_code = ord('_')
        # else:
        #     raise ValueError('error')
        char_code = char_idx + ord(' ')
        text.append(chr(char_code))
    return "".join(text)


def get_next_batch(size=128):
    batch_x = np.zeros([size, IMAGE_WIDTH * IMAGE_HEIGHT])
    batch_y = np.zeros([size, MAX_CAPTCHA* CHAR_SET_LEN])
    for i in range(size):
        text, image = get_train_text_and_image()
        batch_x[i, :] = convert2gray(image).flatten() / 255
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y

def train():
    out = create_cnn()

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(out, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(out, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    # with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=12)) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)
            step += 1

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.8:
                    saver.save(sess, "./crack_capcha.model", global_step=step)
                    break


train()



def crack_captcha():
    output = create_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        # 因为测试集共40个...写的很草率
        count = 0
        for i in range(9896):
            text, image = get_test_text_and_image(i)
            image = convert2gray(image)
            captcha_image = image.flatten() / 255
            text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

            print(text_list)
            predict_text = str(predict_text)
            predict_text = predict_text.replace("[", "").replace("]", "").replace(",", "").replace(" ","")
            if text == predict_text:
                count += 1
                check_result = "，预测结果正确"
            else:
                check_result = "，预测结果不正确"
                print("正确: {}  预测: {}".format(text, predict_text) + check_result)

        # print("正确率:" + str(count) + "/40")
crack_captcha()
