import tensorflow as tf

with tf.device('/gpu:0'):  # Run nodes with GPU 0
    m1 = tf.constant([[3, 5]])
    m2 = tf.constant([[2], [4]])
    product = tf.matmul(m1, m2)

sess = tf.compat.v1.Session()
print(sess.run(product))

sess.close()
