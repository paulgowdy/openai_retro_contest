
# coding: utf-8

# In[1]:


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 

import numpy as np 
import matplotlib.pyplot as plt 


# In[2]:


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[3]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


# In[4]:


sess = tf.InteractiveSession()


# In[5]:


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Dense 1
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dense2
W_fc2 = weight_variable([1024, 64])
b_fc2 = bias_variable([64])

#h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

# Dense3
W_fc3 = weight_variable([64, 4])
b_fc3 = bias_variable([4])

h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3


# In[6]:


E = 50.0 * weight_variable([4, 10])

a = tf.expand_dims(h_fc3,1) # Batch X 1 X 4
b = tf.tile(a, [1,10,1]) # Batch X 10 X 4
c = tf.subtract(b, tf.transpose(E))
d = tf.square(c) # Distances from all cluster centers, in each dim: Batch x 10 x 4

pub_d = tf.sqrt(tf.reduce_sum(d[:,:,:2], axis = 2)) # Batch x 10
pri_d = tf.sqrt(tf.reduce_sum(d[:,:,2:], axis = 2))

big_y_ = tf.tile(tf.expand_dims(y_, 2), [1,1,4])

# Distances from correct cc
e = tf.reduce_sum(tf.multiply(d, big_y_), axis = 1) # Batch x 4


incorrect_mask = 1.0 - big_y_

# Distances from incorrect cc's
f = tf.multiply(d, incorrect_mask) # Batch x 10 x 4

# Incorrect cc hinge loss
g = tf.reduce_sum(tf.maximum(0.0, 1.0 - f), axis = 1) # Batch x 4


#batch_mean_distance_from_correct_cc = tf.reduce_mean(e, axis = 0)
#batch_mean_incorrect_hinge_loss = tf.reduce_mean(g, axis = 0)
h = tf.reduce_mean(e, axis = 0)
i = tf.reduce_mean(g, axis = 0)

pub_loss = tf.reduce_sum(h[:2]) + tf.reduce_sum(i[:2])
pri_loss = tf.reduce_sum(h[2:]) + tf.reduce_sum(i[2:])



combined_cost = pub_loss + pri_loss

pub_grad = tf.gradients(pub_loss, x)
pri_grad = tf.gradients(pri_loss, x)


# In[7]:


pub_correct_prediction = tf.equal(tf.argmin(pub_d,1), tf.argmax(y_,1))
pri_correct_prediction = tf.equal(tf.argmin(pri_d,1), tf.argmax(y_,1))

pub_accuracy = tf.reduce_mean(tf.cast(pub_correct_prediction, tf.float32))
pri_accuracy = tf.reduce_mean(tf.cast(pri_correct_prediction, tf.float32))


# In[16]:


pub_train_step = tf.train.AdamOptimizer(1e-4).minimize(combined_cost)
pri_train_step = tf.train.AdamOptimizer(1e-4).minimize(pri_loss)

#saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())


# In[17]:


color_dict = {0:'deepskyblue',
              1:'navy',
              2:'mediumspringgreen',
              3:'lawngreen',
              4:'mediumvioletred',
              5:'blue',
              6:'purple',
              7:'blueviolet',
              8:'aquamarine',
              9:'magenta'
              }

get_ipython().magic(u'matplotlib notebook')


# In[18]:


def create_adversarial_batch(batch, epsilon = 0.05, max_iterations = 5):
    
    new_samples = np.zeros(batch[0].shape)
    #new_labels = np.zeros(batch[1].shape)
    
    for i in range(batch[0].shape[0]):
        
        
        
        current_label = np.argmax(batch[1][i])
        
        new_label = current_label
        
        while new_label == current_label:
            
            new_label = np.random.randint(0,10)
            
        target_label = [0.0] * 10
        target_label[new_label] += 1.0
        
        #new_labels[i] = np.array(target_label)
        
        new_input = batch[0][i]
        
        attack_iterations = np.random.randint(1,max_iterations)
        
        
        
        for j in range(attack_iterations):
            
            #print(np.array([new_input]).shape, np.array([target_label]).shape)
            
            pub_gradient, pri_gradient = sess.run([pub_grad, pri_grad], feed_dict = {x : np.array([new_input]), y_ : np.array([target_label])})

            #gradient_sign = np.sign(in_gr)

            #new_input = new_input - epsilon * (pub_gradient[0] + pri_gradient[0])
            new_input = new_input - epsilon * (pub_gradient[0])
            new_input = new_input[0]
            
        
        new_samples[i] = new_input
        '''
        pub_distances, pri_distances = sess.run([d1, d2], feed_dict = {x : np.array([new_input])})
        
        print(i, attack_iterations)
        print(current_label, new_label)
        print(np.argmin(pub_distances), np.argmin(pri_distances))
        
        plt.figure()
        plt.imshow(np.reshape(new_input, (28,28)))
        plt.show()
        '''
    return((new_samples, batch[1]))
    
    


# In[ ]:


a = []
b = []


# In[19]:




fig = plt.figure()
plt.ion()

for i in range(10000):
    
    plt.clf()

    batch = mnist.train.next_batch(16)
    
    if i % 2 == 0:
    
        pub_train_step.run(feed_dict = {x : batch[0], y_ : batch[1]})
        
    else:
        
        batch = create_adversarial_batch(batch)
        
        pri_train_step.run(feed_dict = {x : batch[0], y_ : batch[1]})
                      
    if i % 11 == 0:
        
        #print(i)
    
        pub, pri = sess.run([pub_accuracy, pri_accuracy], feed_dict = {x : batch[0], y_ : batch[1]})
    
        print(i, pub, pri)
        a.append(pub)
        b.append(pri)
        
        plt.plot(a)
        plt.plot(b)
        
        fig.canvas.draw()


# In[21]:


ee = E.eval()

print(ee)


# In[22]:


f, (ax1, ax2) = plt.subplots(1,2, figsize = (10,5))

for i in range(10):
    
    ax1.scatter(ee[0][i], ee[1][i], c = color_dict[i])
    
for i in range(10):
    
    ax2.scatter(ee[2][i], ee[3][i], c = color_dict[i])

plt.show()
    


# In[ ]:


batch = mnist.train.next_batch(100)

print(type(batch))
print(type(batch[0]), type(batch[1]))
print(batch[0].shape, batch[1].shape)
print(batch[1][10])


# In[ ]:


batch = mnist.train.next_batch(1000)

publ_d, priv_d = sess.run([pub_d, pri_d], feed_dict = {x : batch[0]})

c = 0

for i in range(1000):
    
    if np.argmin(publ_d[i]) == np.argmin(priv_d[i]):
        
        c += 1

print(c)


# In[23]:


epsilon = 0.1

ad_pair = [9,7]



# In[26]:


target = False

while not target:

    sample_input = mnist.train.next_batch(100)

    if np.argmax(sample_input[1][0]) == ad_pair[0]:

        target = True

target_label = [0.0] * 10
target_label[ad_pair[1]] += 1.0

f, (ax1, ax2) = plt.subplots(1,2, figsize = (10,5))
plt.ion()

for i in range(10):
    
    ax1.scatter(ee[0][i], ee[1][i], c = color_dict[i])
    
for i in range(10):
    
    ax2.scatter(ee[2][i], ee[3][i], c = color_dict[i])

    
    
new_input = sample_input[0]

extra_run = 0

while extra_run < 2:

    #a += 1

    pub_gradient, pri_gradient = sess.run([pub_grad, pri_grad], feed_dict = {x : new_input, y_ : np.array([target_label])})

    #print(new_input.shape)
    #gradient_sign = np.sign(in_gr)

    #new_input = new_input - epsilon * (pub_gradient[0] + pri_gradient[0])
    new_input = new_input - epsilon * (pub_gradient[0])

    new_prediction = pub_d.eval(feed_dict={x:new_input})

    sample_embed = h_fc3.eval(feed_dict={x:new_input})

    ax1.scatter(sample_embed[0][0], sample_embed[0][1], c = 'black', s = 12)
    ax2.scatter(sample_embed[0][2], sample_embed[0][3], c = 'black', s = 12)
    
    if np.argmin(new_prediction[0]) != ad_pair[0]:

        extra_run += 1
    
    f.canvas.draw()
    #plt.draw()
    #plt.pause(0.1)
    
    print(np.argmin(new_prediction[0]), ad_pair[0])
    
    
       




# In[ ]:


c = 0
r = 0
epsilon = 0.15

for p in range(100):
    
    if p % 10 == 0:
        print(p)

    ad_pair = [-1,-1]
    
    while ad_pair[0] == ad_pair[1]:
        
        ad_pair = [np.random.randint(0,10),np.random.randint(0,10)]

    while not target:

        sample_input = mnist.train.next_batch(1)

        if np.argmax(sample_input[1][0]) == ad_pair[0]:

            target = True

    target_label = [0.0] * 10
    target_label[ad_pair[1]] += 1.0

    new_input = sample_input[0]

    extra_run = 0

    while extra_run < 2:

        pub_gradient, pri_gradient = sess.run([pub_grad, pri_grad], feed_dict = {x : new_input, y_ : np.array([target_label])})

        #print(new_input.shape)
        #gradient_sign = np.sign(in_gr)

        #new_input = new_input - epsilon * (pub_gradient[0] + pri_gradient[0])
        new_input = new_input - epsilon * (pub_gradient[0])

        new_prediction = pub_d.eval(feed_dict={x:new_input})

        sample_embed = h_fc3.eval(feed_dict={x:new_input})

        if np.argmin(new_prediction[0]) == ad_pair[1]:
        #if np.argmin(new_prediction[0]) != ad_pair[0]:

            extra_run += 1

    publ_d, priv_d = sess.run([pub_d, pri_d], feed_dict = {x : new_input})

    #print(np.argmin(publ_d[0]), np.argmin(priv_d[0]))

    if np.argmin(publ_d[0]) != np.argmin(priv_d[0]):

        c += 1
        
    if np.argmin(priv_d[0]) == ad_pair[0]:
        r += 1

print('--')
print(c, r)

