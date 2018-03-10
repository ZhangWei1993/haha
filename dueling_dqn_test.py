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
    pickle.dump(memory, open("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/four/memory/save_memory_"+str(episode)+".txt", 'wb'))
    np.save("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/four/memory/save_epsilon"+str(episode)+".npy",epsilon)
#save the memory reply data after a number of episode in case of the training process break down
def load_memory(episode):
    memory=pickle.load(open("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/four/memory/save_memory_"+str(episode)+".txt", 'rb'))
    ep = np.load("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/four/memory/save_epsilon"+str(episode)+".npy")
    return memory, ep

if __name__ == "__main__":
    env = gym.make('GazeboRoom1TurtlebotLidar-v0')
    #env = env.unwrapped
    #env.seed(1)
    MEMORY_SIZE = 10000
    ACTION_SPACE = 3
#    train_steps = 1000000
    start = 4500
    EPISODES = 60000
    state_size = 32
    treward = np.zeros(60000)
    sess = tf.Session()
    #with tf.variable_scope('natural'):
    #    natural_DQN = DuelingDQN(
    #        n_actions=ACTION_SPACE, e_greedy = 0.99, n_features=50, memory_size=MEMORY_SIZE,
    #        e_greedy_increment=0.99/train_steps, sess=sess, dueling=False)
    
    with tf.variable_scope('dueling'):
        dueling_DQN = DuelingDQN(
            action_size=ACTION_SPACE, state=state_size, epsilon=0.00000001, memory_size=MEMORY_SIZE,
            e_greedy_decay=0.9999, sess=sess, dueling=True, cnn = True, fcn=None, output_graph=True)    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    '''
    reload parameters
#    saver.restore(sess,"/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/third/"+str(start)+".ckpt")#reload parameters
#    dueling_DQN.memory,ep = load_memory(episode=25500)#reload memory
#    dueling_DQN.epsilon = float(ep)    '''
    for star in range(0,6000,500):
        saver.restore(sess,"/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/four/"+str(start)+".ckpt")#reload parameters
    #    dueling_DQN.memory,ep = load_memory(episode=2000)#reload memory
    #    dueling_DQN.epsilon = float(ep)
        for e in range(start, start+2):
            state,done = env.reset()
            next_state, reward, done, _ = env.step(random.randrange(3))
            next_state, reward, done, _ = env.step(random.randrange(3))
            state = next_state
    #        state = tf.image.per_image_standardization(state)
            state = np.reshape(state,(1,48,64))
            state = np.stack((state,state),axis=3)  
            total_reward = 0
            for time in range(1500):
                # env.render()
                action = dueling_DQN.act(state,"boltzmann")
                next_state, reward, done, _ = env.step(action)               
    #            next_state = tf.image.per_image_standardization(next_state)
                next_state1 = np.reshape(next_state,(1,48,64))
                all_same = np.stack((next_state1,next_state1),axis=3)
#                dueling_DQN.plot_image(all_same,kind = 'all_same') 
                next_state = np.reshape(next_state,(1,48,64,1))
                next_state = np.append(next_state,state[:,:,:,:1],axis=3)
#                next_state = np.stack((next_state,next_state,next_state),axis=3)
    #            reward = reward if not done else -10
    #            next_state = np.reshape(next_state, [1, state_size])
    #            dueling_DQN.remember(state, action, reward, next_state,done)
                state = next_state
#                dueling_DQN.plot_image(state,kind = 'sucessive')
                total_reward = reward + total_reward
    #            print(action)
    #            if time == 1499:
    #                done = True
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                          .format(e, EPISODES, total_reward, dueling_DQN.epsilon))
                    break
    #            if len(dueling_DQN.memory) == MEMORY_SIZE:
    #                dueling_DQN.learn()
            treward[e] = total_reward
#        if len(dueling_DQN.memory) == MEMORY_SIZE:        
#            dueling_DQN.epsilon_decay_per_epoch()
#        if e % 500 == 0:            
#            save_path = saver.save(sess, "/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/third/"+str(e)+".ckpt")
#            np.save('total_reward.npy',treward)
#        if total_reward > -5:
#            save_path = saver.save(sess, "/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/third/good/"+str(e)+".ckpt")
#        if e % 5000 == 0:
#            save_memory(dueling_DQN.memory,e,dueling_DQN.epsilon)
            
            
            
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

