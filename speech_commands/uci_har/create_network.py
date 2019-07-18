import tensorflow as tf

def create_model(data_x,no_of_layers=3,WEIGHT_DECAY=1e-4,isTraining=False,NUM_CLASSES=2):
    
   with tf.variable_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(
            inputs=data_x,
            filters=64,
            kernel_size=[1, 32],
            padding='VALID',
            use_bias=False,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        )
        #FIXME
        bn1 = tf.layers.batch_normalization(conv1, training=isTraining)
        relu1 = tf.nn.relu(bn1)
     
   with tf.variable_scope('conv2') as scope:
        conv2 = tf.layers.conv2d(
            inputs=relu1,
            filters=128,
            kernel_size=[1, 16],
            padding='VALID',
            use_bias=False,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        )
        #FIXME
        bn2 = tf.layers.batch_normalization(conv2, training=isTraining)
        relu2 = tf.nn.relu(bn2)
        # pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')
   with tf.variable_scope('conv3') as scope:
        conv3 = tf.layers.conv2d(
            inputs=relu2,
            filters=256,
            kernel_size=[1, 8],
            padding='VALID',
            use_bias=False,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        )
   with tf.variable_scope('conv4') as scope:
        conv4 = tf.layers.conv2d(
            inputs=relu2,
            filters=256,
            kernel_size=[1, 4],
            padding='VALID',
            use_bias=False,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        )
        
   if no_of_layers==3:
        with tf.variable_scope('fully_connected') as scope:
            flat = tf.layers.flatten(relu3)
       
            logits = tf.layers.dense(inputs=flat, units=NUM_CLASSES, name=scope.name, use_bias=False, kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))
   if no_of_layers==2:
        with tf.variable_scope('fully_connected') as scope:
            flat = tf.layers.flatten(relu2)
       
            logits = tf.layers.dense(inputs=flat, units=NUM_CLASSES, name=scope.name, use_bias=False, 
kernel_regularizer=tf.contrib.layers.l2_regularizer(WEIGHT_DECAY))
   return logits