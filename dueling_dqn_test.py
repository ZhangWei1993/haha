"""
Dueling DQN & Natural DQN comparison

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from Obstacle_avoidance import DuelingDQN2
import numpy as np
import matplotlib.pyplot as plt
import random
import gym_gazebo
import pickle
import tensorflow as tf


def save_memory(memory,episode,epsilon):
    pickle.dump(memory, open("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/obstacle/memory/save_memory_"+str(episode)+".txt", 'wb'))
    np.save("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/obstacle/memory/save_epsilon"+str(episode)+".npy",epsilon)
#save the memory reply data after a number of episode in case of the training process break down
def load_memory(episode):
    memory=pickle.load(open("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/obstacle/memory/save_memory_"+str(episode)+".txt", 'rb'))
    ep = np.load("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/obstacle/memory/save_epsilon"+str(episode)+".npy")
    return memory, ep

if __name__ == "__main__":
    env = gym.make('GazeboRoom1TurtlebotLidar-v0')
    #env = env.unwrapped
    #env.seed(1)
    MEMORY_SIZE = 10000
    ACTION_SPACE = 5
#    train_steps = 1000000
    start = 10500
    EPISODES = 60000
    state_size = 100
    treward = np.zeros(60000)
    sess = tf.Session()
    #with tf.variable_scope('natural'):
    #    natural_DQN = DuelingDQN(
    #        n_actions=ACTION_SPACE, e_greedy = 0.99, n_features=50, memory_size=MEMORY_SIZE,
    #        e_greedy_increment=0.99/train_steps, sess=sess, dueling=False)
    
    with tf.variable_scope('obstacle_avoiding'):
        dueling_DQN = DuelingDQN2(
            action_size=ACTION_SPACE, rate=1.0,state=state_size, epsilon=0.0, memory_size=MEMORY_SIZE,
            e_greedy_decay=0.9995, sess=sess, dueling=True, cnn = True, fcn=None, output_graph=True)    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    '''
    reload parameters
    saver.restore(sess,"/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/for/"+str(start)+".ckpt")#reload parameters
#    dueling_DQN.memory,ep = load_memory(episode=25500)#reload memory
    dueling_DQN.epsilon = float(ep)    '''
    saver.restore(sess,"/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/obstacle/"+str(10500)+".ckpt")#reload parameters
#    dueling_DQN.memory,ep = load_memory(episode=2000)#reload memory
#    dueling_DQN.epsilon = float(ep)
    for e in range(start, EPISODES):
        state, target, done = env.reset()
        target = np.reshape(target,(1,2)).astype(np.float32)
        state = np.reshape(state,(1,100))
#        state = np.stack((state,state),axis=3)      
        old_action = 1
        total_reward = 0
        for time in range(500):
            # env.render()
            reward_plus = 0
            action = dueling_DQN.act(state,target,"e_greedy")
            if action == old_action:
                if action == 3 or action ==4:
                    reward_plus = -5
#            dueling_DQN.plot_image(state,kind = 'sucessive')
            next_state, reward, next_target, vel_state, done = env.step(action)
            next_target = np.reshape(next_target,(1,2)).astype(np.float32)
#            if target[0,0]<=0.2:
#                done = True
#            reward = reward+ 0.1*(-next_target[0,0]+target[0,0])
#            print("distance:{},angle:{}".format(next_target[0,0],next_target[0,1]))
#            next_state = tf.image.per_image_standardization(next_state)
            next_state = np.reshape(next_state,(1,100))
            reward = reward_plus + reward
#            next_state = np.append(next_state,state[:,:,:,:1],axis=3)
#            reward = reward if not done else -10
#            next_state = np.reshape(next_state, [1, state_size])
            dueling_DQN.remember(state, target, action, reward, next_state,next_target,done)
            state = next_state
            target = next_target
            total_reward = reward + total_reward
            old_action = action
#            print(action)
#            if time == 1499:
#                done = True
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, total_reward, dueling_DQN.epsilon))
                break
            if len(dueling_DQN.memory) == MEMORY_SIZE:
                dueling_DQN.learn()
#        treward[e] = total_reward

            
            
            
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

