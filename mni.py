#-*- coding:utf-8 -*-
import input_data
import tensorflow as tf

# mnistデータを格納したオブジェクトを呼び出す
mnist = input_data.read_data_sets("data/", one_hot=True)

# 全訓練データの取得

# 訓練用の入力データ、正解データをミニバッチ数を指定して取得
train_images, train_labels = mnist.train.next_batch(50)

# テスト用の全画像データを取得
test_images = mnist.test.images
# テスト用の全正解データを取得
test_labels = mnist.test.labels

# 入力データを定義
x = tf.placeholder(tf.float32, [None, 784])
# 入力画像をログに出力
img = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image("input_data", img, 10)

# 畳み込み層１
f1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
conv1 = tf.nn.conv2d(img, f1, strides=[1, 1, 1, 1], padding='SAME')
b1 = tf.Variable(tf.constant(0.1, shape=[32]))
h_conv1 = tf.nn.relu(conv1 + b1)

# プーリング層1
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 畳み込み層2
f2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
conv2 = tf.nn.conv2d(h_pool1, f2, strides=[1, 1, 1, 1], padding='SAME')
b2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(conv2+b2)

# プーリング層2
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 畳み込まれているものをフラットな形に変換
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# 全結合層
w_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# 出力層
w_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
out = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

# 入力層から中間層
##with tf.name_scope("hidden"):
    ##w_1 = tf.Variable(tf.truncated_normal([784, 64], stddev=0.1), name="w1")
    ##b_1 = tf.Variable(tf.zeros([64]), name="b1")
    ##h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)

    # 中間層の重みの分布をログ出力
    ##tf.summary.histogram('w_1', w_1)

# 中間層から出力層
##with tf.name_scope("output"):
    ##w_2 = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1), name="w2")
    ##b_2 = tf.Variable(tf.zeros([10]), name="b2")
    ##out = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2)

y = tf.placeholder(tf.float32, [None, 10])
# 誤差関数
##with tf.name_scope("loss"):
    # loss = tf.reduce_mean(tf.square(y - out))
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(out + 1e-5), axis=[1]))
    ##tf.summary.scalar("loss", loss)

# 訓練
##with tf.name_scope("train"):
    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 評価
##with tf.name_scope("accuracy"):
correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    ##tf.summary.scalar("accuracy", accuracy)

##summary_op = tf.summary.merge_all()

# 初期化
init = tf.global_variables_initializer()

# 実行
with tf.Session() as sess:

    sess.run(init)
    ##summary_writer = tf.summary.FileWriter("logs", sess.graph)

    # テストデータをロード
    test_images = mnist.test.images
    test_labels = mnist.test.labels

    for i in range(1000):
        step = i + 1
        train_images, train_labels = mnist.train.next_batch(50)
        sess.run(train_step, feed_dict={x:train_images, y:train_labels})

        if step % 100 == 0:
            acc_val = sess.run(accuracy, feed_dict={x:test_images, y:test_labels})
            print('Step %d: accuracy = %.2f' % (step, acc_val))
            # ログを取る処理を実行する
            ##summary_str = sess.run(summary_op, feed_dict={x:test_images, y:test_labels})
            # ログ情報のプロトコルバッファを書き込む
            ##summary_writer.add_summary(summary_str, step)
