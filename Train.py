import tensorflow as tf

def train():
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros[784, 10])
    b = tf.Variable(tf.zeros[10])
    return

def conv_net(x, W, b, dropout):
    # Reshape input
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    return

def main():
    x = [[1,2,3],[4,5,6],[7,8,9]]
    x = tf.reshape(x, shape=[-1])
    return

main()