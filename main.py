import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# define parameters
mnist = input_data.read_data_sets('MNIST_data/', one_hot= True)

batch_size = 128
epsilon = 0.000001
iteration_num = 3
lambda_v = 0.5
reg_v = 0.0001
m_p = 0.9
m_m = 0.1


def routing(input,b_IJ):
    b_ij = b_IJ

    W = tf.get_variable('Weights', shape = (1,1152,10,8,16),dtype=tf.float32, initializer=tf.random_normal_initializer())
    ipt = tf.tile(input,[1,1,10,1,1])
    W = tf.tile(W,[batch_size,1,1,1,1])


    u_hat = tf.matmul(W,ipt,transpose_a=True)

    # print('u_hat.size ===>',u_hat.get_shape)

    for i in range(iteration_num):
        with tf.variable_scope('iter_' + str(iteration_num)):
            print('routing in progress ===>' , i)
            c_ij = tf.nn.softmax(b_ij,dim=2)

            print('c_ij.size ===>',c_ij.get_shape())

            s_j = tf.multiply(c_ij, u_hat)
            s_j = tf.reduce_sum(s_j,axis = 1, keep_dims=True)
            v_j = squash(s_j)
            v_j_tiled = tf.tile(v_j, [1,1152,1,1,1])
            u_produce_v = tf.matmul(u_hat,v_j_tiled, transpose_a = True)
            if i < iteration_num -1:
                b_ij = b_ij + u_produce_v

    return v_j






def squash(vec):

    vec_sq_norm = tf.reduce_sum(tf.square(vec),-2,keep_dims=True)
    scalar_factor = vec_sq_norm/(1 + vec_sq_norm)/tf.sqrt(vec_sq_norm + epsilon)
    vec_squashed = scalar_factor * vec
    return vec_squashed



x = tf.placeholder(tf.float32,[batch_size,28,28,1])
y = tf.placeholder(tf.float32,[batch_size,10,1])

network = tf.contrib.layers.conv2d(x,num_outputs=256,kernel_size=9,stride=1,padding='VALID')

caps = []
for i in range(8):
    with tf.variable_scope('ConvUnit_' + str(i)):
        caps_i = tf.contrib.layers.conv2d(network,num_outputs=32,kernel_size=9,stride=2,padding='VALID',activation_fn = tf.nn.relu)

        caps_i = tf.reshape(caps_i, shape=(batch_size,-1,1,1))

        caps.append(caps_i)

capsules = tf.concat(caps, axis = 2)



primary_cap = squash(capsules)

# print('squash',primary_cap.get_shape()) correct


b_ij = tf.constant(np.zeros([batch_size,primary_cap.shape[1].value,10,1,1]),dtype=np.float32)
reshaped_cap = tf.reshape(primary_cap,shape=(batch_size,-1,1,primary_cap.shape[-2].value,1))

caps_f = routing(reshaped_cap,b_ij)
caps_o = tf.squeeze(caps_f,axis =1)

print('caps_o',caps_o.shape)

with tf.variable_scope('Masking'):
    vec_length = tf.sqrt(tf.reduce_sum(tf.square(caps_o),axis = 2,keep_dims = True) + epsilon)
    masked_vec = tf.multiply(tf.squeeze(caps_o),tf.reshape(y,(-1,10,1)))


# with tf.variable_scope('Decoder'):
#     vec_j = tf.reshape(masked_vec,shape=(batch_size,-1))
#     fc1 = tf.contrib.layers.fully_connected(vec_j,num_outputs = 512)
#
#     print('fc1 ===>',fc1.get_shape())
#
#     fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
#
#     print('fc2 ===>', fc2.get_shape())
#
#     fc_decode = tf.contrib.layers.fully_connected(fc2, num_outputs = 784,activation_fn = tf.sigmoid)


max_l = tf.square(tf.maximum(0., m_p - vec_length))
max_r = tf.square(tf.maximum(0., vec_length - m_m))



max_l = tf.reshape(max_l, shape = (batch_size,-1))
max_r = tf.reshape(max_r, shape = (batch_size,-1))


T_c = tf.reshape(y,(batch_size,10))


L_c = T_c * max_l + lambda_v * (1-T_c) * max_r

margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis = 1))

# origin_image = tf.reshape(x,shape=(batch_size,-1))
# squared = tf.square(fc_decode - origin_image)
# reconstruct_error = tf.reduce_mean(squared)

total_loss =  margin_loss

optmizer = tf.train.AdamOptimizer()
train_op = optmizer.minimize(total_loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

test_image = mnist.test.images[0:batch_size]
test_image = test_image.reshape(batch_size,28,28,1)

reconstruct_result = sess.run(vec_length,feed_dict={x:test_image})
# print(reconstruct_result)


for iter in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs =batch_xs.reshape(batch_size,28,28,1)
    batch_ys = batch_ys.reshape(batch_size,10,1)
    _, total_l = sess.run([train_op,total_loss],feed_dict={x:batch_xs,y:batch_ys})




    # print(batch_ys[0])
    print(iter,'===>',total_l)

    # classification_result = sess.run(vec_length,feed_dict={x:test_image})

    # print(classification_result.shape)
    # print(classification_result[0])










