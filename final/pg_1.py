import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
from skimage import transform

from retro_contest.local import make

from sonic_util import SonicDiscretizer

env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
env = SonicDiscretizer(env)

print('action space:', env.action_space.n)

# Environment Parameters
n_actions = env.action_space.n
n_epochs = 5000
n = 0
average = []
step = 1
batch_size = 5000
render = True

# Define our three actions of moving forward, turning left & turning right
buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
           ['DOWN', 'B'], ['B']]
choice = []

for action in actions:
    arr = [0] * 12

    for button in action:

        arr[buttons.index(button)] =1

    choice.append(arr)

#print(choice)


# Hyper Parameters
alpha = 1e-4
gamma = 0.99
normalize_r = True
save_path='pg.ckpt'
value_scale = 0.5
entropy_scale = 0.00
gradient_clip = 40

# Apply discount to episode rewards & normalize
def discount(r, gamma, normal):
    discount = np.zeros_like(r)
    G = 0.0
    for i in reversed(range(0, len(r))):
        G = G * gamma + r[i]
        discount[i] = G
    # Normalize
    if normal:
        mean = np.mean(discount)
        std = np.std(discount)
        discount = (discount - mean) / (std)
    return discount

# Conv Layers
convs = [16,32]
kerns = [8,8]
strides = [4,4]
pads = 'valid'
fc = 256
activ = tf.nn.elu

# Function for resizing image
def resize(image):
    # Greyscale Image
    x = np.mean(image,-1)
    # Normalize Pixel Values
    x = x/255
    x = transform.resize(x, [84,84])
    return(x)

# Tensorflow Variables
X = tf.placeholder(tf.float32, (None,84,84,1), name='X')
Y = tf.placeholder(tf.int32, (None,), name='actions')
R = tf.placeholder(tf.float32, (None,), name='reward')
N = tf.placeholder(tf.float32, (None), name='episodes')
D_R = tf.placeholder(tf.float32, (None,), name='discounted_reward')

# Policy Network
conv1 = tf.layers.conv2d(
        inputs = X,
        filters = convs[0],
        kernel_size = kerns[0],
        strides = strides[0],
        padding = pads,
        activation = activ,
        name='conv1')

conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters = convs[1],
        kernel_size = kerns[1],
        strides = strides[1],
        padding = pads,
        activation = activ,
        name='conv2')

flat = tf.layers.flatten(conv2)

dense = tf.layers.dense(
        inputs = flat,
        units = fc,
        activation = activ,
        name = 'fc')

logits = tf.layers.dense(
         inputs = dense,
         units = n_actions,
         name='logits')

value = tf.layers.dense(
        inputs=dense,
        units = 1,
        name='value')

calc_action = tf.multinomial(logits, 1)
aprob = tf.nn.softmax(logits)
action_logprob = tf.nn.log_softmax(logits)


tf.trainable_variables()

def rollout(batch_size, render):

    states, actions, rewards, rewardsFeed, discountedRewards = [], [], [], [], []
    state = resize(env.reset())
    episode_num = 1
    action_repeat = 3
    reward = 0

    while True:

        if render:
            env.render()

        # Run State Through Policy & Calculate Action
        feed = {X: state.reshape(1, 84, 84, 1)}
        action = sess.run(calc_action, feed_dict=feed)
        action = action[0][0]
        #print('action', action)

        # Perform Action
        for i in range(action_repeat):
            state2, reward2, done, info = env.step(choice[action])
            #print(done)
            reward += reward2
            if done:
                break

        #print(reward)
        # Store Results
        states.append(state)
        rewards.append(reward)
        actions.append(action)

        # Update Current State
        reward = 0
        state = resize(state2)

        if done:
            # Track Discounted Rewards
            rewardsFeed.append(rewards)
            discountedRewards.append(discount(rewards, gamma, normalize_r))

            #episode_num += 1
            #print(len(np.concatenate(rewardsFeed)))

            if len(np.concatenate(rewardsFeed)) > batch_size:
                break

            # Reset Environment
            rewards = []
            state = resize(env.reset())
            episode_num += 1


    return np.stack(states), np.stack(actions), np.concatenate(rewardsFeed), np.concatenate(discountedRewards), episode_num

mean_reward = tf.divide(tf.reduce_sum(R), N)

# Define Losses
pg_loss = tf.reduce_mean((D_R - value) * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
value_loss = value_scale * tf.reduce_mean(tf.square(D_R - value))
entropy_loss = -entropy_scale * tf.reduce_sum(aprob * tf.exp(aprob))
loss = pg_loss + value_loss - entropy_loss

# Create Optimizer
optimizer = tf.train.AdamOptimizer(alpha)
grads = tf.gradients(loss, tf.trainable_variables())
grads, _ = tf.clip_by_global_norm(grads, gradient_clip) # gradient clipping
grads_and_vars = list(zip(grads, tf.trainable_variables()))
train_op = optimizer.apply_gradients(grads_and_vars)

# Initialize Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

while step < n_epochs+1:
    # Gather Training Data
    print('Epoch', step)
    s, a, r, d_r, n = rollout(batch_size,render)
    print('rollout:', n)
    mean_reward = np.sum(r)/n
    average.append(mean_reward)
    print('Training Episodes: {}  Average Reward: {:4.2f}  Total Average: {:4.2f}'.format(n, mean_reward, np.mean(average)))

    # Update Network
    sess.run(train_op, feed_dict={X:s.reshape(len(s),84,84,1), Y:a, D_R: d_r})

    # Write TF Summaries
    #summary = sess.run(write_op, feed_dict={X:s.reshape(len(s),84,84,1), Y:a, D_R: d_r, R: r, N:n})
    #writer.add_summary(summary, step)
    #writer.flush()

    # Save Model
    if step % 10 == 0:
          print("SAVED MODEL")
          saver.save(sess, save_path, global_step=step)

    step += 1
