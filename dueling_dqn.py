"""
Dueling DQN & Natural DQN comparison

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DuelingDQN
import numpy as np
import matplotlib.pyplot as plt
import random
import gym_gazebo
import pickle
import tensorflow as tf


def save_memory(memory,episode,epsilon):
    pickle.dump(memory, open("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/six/memory/save_memory_"+str(episode)+".txt", 'wb'))
    np.save("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/six/memory/save_epsilon"+str(episode)+".npy",epsilon)
#save the memory reply data after a number of episode in case of the training process break down
def load_memory(episode):
    memory=pickle.load(open("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/six/memory/save_memory_"+str(episode)+".txt", 'rb'))
    ep = np.load("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/six/memory/save_epsilon"+str(episode)+".npy")
    return memory, ep

if __name__ == "__main__":
    env = gym.make('GazeboRoom1TurtlebotLidar-v0')
    #env = env.unwrapped
    #env.seed(1)
    MEMORY_SIZE = 10000
    ACTION_SPACE = 5
#    train_steps = 1000000
    start = 0
    EPISODES = 80000
    state_size = 32
    treward = np.zeros(60000)
    sess = tf.Session()
    #with tf.variable_scope('natural'):
    #    natural_DQN = DuelingDQN(
    #        n_actions=ACTION_SPACE, e_greedy = 0.99, n_features=50, memory_size=MEMORY_SIZE,
    #        e_greedy_increment=0.99/train_steps, sess=sess, dueling=False)
    
    with tf.variable_scope('dueling'):
        dueling_DQN = DuelingDQN(
            action_size=ACTION_SPACE, state=state_size, epsilon=1.0, memory_size=MEMORY_SIZE,
            e_greedy_decay=0.999, sess=sess, dueling=True, cnn = True, fcn=None, output_graph=True,is_train = True)    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    '''
    reload parameters
    saver.restore(sess,"/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/six/"+str(start)+".ckpt")#reload parameters
#    dueling_DQN.memory,ep = load_memory(episode=25500)#reload memory
    dueling_DQN.epsilon = float(ep)    '''
#    saver.restore(sess,"/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/six/"+str(start)+".ckpt")#reload parameters
#    dueling_DQN.memory,ep = load_memory(episode=3000)#reload memory
#    dueling_DQN.epsilon = ep
    for e in range(start, EPISODES):
        state,done = env.reset()
        next_state, reward, done, _ = env.step(random.randrange(5))
        next_state, reward, done, _ = env.step(random.randrange(5))
        state = next_state
        state = np.reshape(state,(48,64,1))
#        state = np.stack((state,state),axis=3)        
        total_reward = 0
#        done_time = 0
#        action_1 = 2;
#        action_2 = 2;
        for time in range(1500):
            # env.render()
            action = dueling_DQN.act(state,"boltzmann")
#            action_2 = action_1
#            action_1 = action
#            dueling_DQN.plot_image(state,kind = 'sucessive')
            next_state, reward, done, _ = env.step(action=action)            
#            next_state = tf.image.per_image_standardization(next_state)
            next_state = np.reshape(next_state,(48,64,1))
#            next_state = np.append(next_state,state[:,:,:,:1],axis=3)
#            reward = reward if not done else -10
#            next_state = np.reshape(next_state, [1, state_size])
            dueling_DQN.remember(state, action, reward, next_state,done)
            state = next_state
            total_reward = reward + total_reward
#            print(action)
#            if time == 1499:
#                done = True
            if done:
#                done_time = done_time + 1             
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, total_reward, dueling_DQN.epsilon))
                break
#                total_reward = total_reward - reward
#                next_state, reward, done, _ = env.step(action_1+5)
#                next_state, reward, done, _ = env.step(action_2+5)
#                state = next_state
#if the robot crashes for 5 times, break            
#            if done_time == 5:
#                break
            if len(dueling_DQN.memory) == MEMORY_SIZE:
                dueling_DQN.learn()
        treward[e] = total_reward
        if len(dueling_DQN.memory) == MEMORY_SIZE:        
            dueling_DQN.epsilon_decay_per_epoch()
        if e % 500 == 0:            
            save_path = saver.save(sess, "/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/six/"+str(e)+".ckpt")
            np.save('total_reward.npy',treward)
        if total_reward > -5:
            save_path = saver.save(sess, "/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/six/good/"+str(e)+".ckpt")
        if e % 1500 == 0:
            save_memory(dueling_DQN.memory,e,dueling_DQN.epsilon)
            
            
            
#    c_dueling, r_dueling = train(dueling_DQN)
#
#    plt.figure(1)
#    #plt.plot(np.array(c_natural), c='r', label='natural')
#    plt.plot(np.array(c_dueling), c='b', label='dueling')
#    plt.legend(loc='best')
#    plt.ylabel('cost')
#    plt.xlabel('training steps')
#    plt.grid()
#    
#    plt.figure(2)
#    #plt.plot(np.array(r_natural), c='r', label='natural')
#    plt.plot(np.array(r_dueling), c='b', label='dueling')
#    plt.legend(loc='best')
#    plt.ylabel('accumulated reward')
#    plt.xlabel('training steps')
#    plt.grid()
#    
#    plt.show()

