import tensorflow as tf
class text_cnn(tf.keras.Model):
    def __init__(self):
        super(text_cnn, self).__init__()
        self.c1 = tf.keras.layers.Conv1D(filters=64, kernel_size=(3), padding='VALID')  # 卷积层1
        self.b1 = tf.keras.layers.BatchNormalization()  # BN层1
        self.a1 = tf.keras.layers.Activation('relu')  # 激活层1
        self.p1 = tf.keras.layers.MaxPool1D(pool_size=(3), strides=2, padding='VALID')
        self.d1 = tf.keras.layers.Dropout(0.3)  # dropout层

        self.c2 = tf.keras.layers.Conv1D(filters=64, kernel_size=(4), padding='VALID')
        self.b2 = tf.keras.layers.BatchNormalization()  # BN层1
        self.a2 = tf.keras.layers.Activation('relu')  # 激活层1
        self.p2 = tf.keras.layers.MaxPool1D(pool_size=(4), strides=2, padding='VALID')
        self.d2 = tf.keras.layers.Dropout(0.3)  # dropout层

        self.c3 = tf.keras.layers.Conv1D(filters=64, kernel_size=(5), padding='VALID')
        self.b3 = tf.keras.layers.BatchNormalization()  # BN层1
        self.a3 = tf.keras.layers.Activation('relu')  # 激活层1
        self.p3 = tf.keras.layers.MaxPool1D(pool_size=(5), strides=2, padding='VALID')
        self.d3 = tf.keras.layers.Dropout(0.3)  # dropout层

        self.flatten = tf.keras.layers.Flatten()

    def call(self, x):
        x1 = self.c1(x)
        x1 = self.b1(x1)
        x1 = self.a1(x1)
        x1 = self.p1(x1)
        x1 = self.d1(x1)
        x1 = self.flatten(x1)

        x2 = self.c2(x)
        x2 = self.b2(x2)
        x2 = self.a2(x2)
        x2 = self.p2(x2)
        x2 = self.d2(x2)
        x2 = self.flatten(x2)

        x3 = self.c3(x)
        x3 = self.b3(x3)
        x3 = self.a3(x3)
        x3 = self.p3(x3)
        x3 = self.d3(x3)
        x3 = self.flatten(x3)

        y = tf.concat([x1, x2, x3], 1)
        return y