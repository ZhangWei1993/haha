"""
Dueling DQN & Natural DQN comparison

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from target import DuelingDQN
from Obstacle_avoidance import DuelingDQN2
from High_level import HDueling
import numpy as np
import matplotlib.pyplot as plt
import random
import gym_gazebo
import pickle
import tensorflow as tf


def save_memory(memory,episode,epsilon):
    pickle.dump(memory, open("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/HRL/memory/save_memory_"+str(episode)+".txt", 'wb'))
    np.save("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/HRL/memory/save_epsilon"+str(episode)+".npy",epsilon)
#save the memory reply data after a number of episode in case of the training process break down
def load_memory(episode):
    memory=pickle.load(open("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/HRL/memory/save_memory_"+str(episode)+".txt", 'rb'))
    ep = np.load("/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/HRL/memory/save_epsilon"+str(episode)+".npy")
    return memory, ep

if __name__ == "__main__":
    env = gym.make('GazeboRoom1TurtlebotLidar-v0')
    #env = env.unwrapped
    #env.seed(1)
    MEMORY_SIZE = 10000
    ACTION_SPACE = 5
#    train_steps = 1000000
    start = 0
    EPISODES = 60000
    state_size = 12
    treward = np.zeros(60000)
    #first graph for obstacle avoidance
    g1 = tf.Graph()
    sess1 = tf.Session(graph = g1)
    with sess1.as_default(): 
        with g1.as_default():
            with tf.variable_scope('dueling'):
                dueling_DQN1 = DuelingDQN(
                    action_size=ACTION_SPACE, state=100, epsilon=0.0, memory_size=MEMORY_SIZE,
                    e_greedy_decay=0.9992, sess=sess1, dueling=True, cnn = True, fcn=None, output_graph=True) 
            sess1.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(sess1,"/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/laser/target/"+str(4000)+".ckpt")#reload parameters
    #second graph for target reaching    
    g2 = tf.Graph()
    sess2 = tf.Session(graph = g2)
    with sess2.as_default(): 
        with g2.as_default():
            with tf.variable_scope('obstacle_avoiding'):
                dueling_DQN2 = DuelingDQN2(
                    action_size=ACTION_SPACE, state=100, epsilon=0.0, memory_size=MEMORY_SIZE,
                    e_greedy_decay=0.9992, sess=sess2, dueling=True, cnn = True, fcn=None, output_graph=True) 
            sess2.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(sess2,"/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/obstacle/"+str(4000)+".ckpt")#reload parameters
        #second graph for target reaching    
    g3 = tf.Graph()
    sess3 = tf.Session(graph = g3)
    with sess3.as_default(): 
        with g3.as_default():
            with tf.variable_scope('high_level'):
                dueling_DQN3 = HDueling(
                    action_size=ACTION_SPACE, state=state_size, epsilon=1.0, memory_size=MEMORY_SIZE,
                    e_greedy_decay=0.9992, sess=sess3, dueling=True, cnn = True, fcn=None, output_graph=True) 
            sess3.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)
#            saver.restore(sess3,"/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/HRL/"+str(4000)+".ckpt")#reload parameters
    #with tf.variable_scope('natural'):
    #    natural_DQN = DuelingDQN(
    #        n_actions=ACTION_SPACE, e_greedy = 0.99, n_features=50, memory_size=MEMORY_SIZE,
    #        e_greedy_increment=0.99/train_steps, sess=sess, dueling=False)
    
    '''
    reload parameters
    saver.restore(sess,"/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/for/"+str(start)+".ckpt")#reload parameters
#    dueling_DQN.memory,ep = load_memory(episode=25500)#reload memory
    dueling_DQN.epsilon = float(ep)    '''
#    saver.restore(sess,"/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/for/"+str(4000)+".ckpt")#reload parameters
#    dueling_DQN.memory,ep = load_memory(episode=3000)#reload memory
#    dueling_DQN.epsilon = float(ep)
    alpha = 1
    for e in range(start, EPISODES):
        alpha = alpha*0.9
        print("alpha:{}".format(alpha))
        raw_state, target, done = env.reset()
        target = np.reshape(target,(1,2)).astype(np.float32)
        raw_state = np.reshape(raw_state,(1,100))
        with sess2.as_default():
            with sess2.graph.as_default():  #
                raw_action2,low_act2 = dueling_DQN2.output(raw_state,target)
        with sess1.as_default():
            with sess1.graph.as_default():  #
                raw_action1,low_act1 = dueling_DQN1.output(target)
        state = np.concatenate((raw_action1,raw_action2),axis=1)
        state = np.reshape(state,(1,12))
#        state = np.stack((state,state),axis=3)    
        total_reward = 0
        for time in range(500):
            # env.render()
            with sess3.as_default():
                with sess3.graph.as_default():
                    if np.random.rand() > dueling_DQN3.epsilon:               
                        action = dueling_DQN3.act(state,'deter')
#                    elif target[0,0] <  1:
#                        action = low_act1
                    else:
                        #randomly select the action
#                        action = np.random.choice([low_act1,low_act2])
                        #select the maximum sum
#                        act1 = np.max(state[0,0:6])/(np.max(state[0,0:6])-np.min(state[0,0:6]))
#                        act2 = np.max(state[0,6:12])/(np.max(state[0,6:12])-np.min(state[0,6:12]))
#                        if act1>act2:
#                            action = low_act1
#                        else:
#                            action = low_act2
#                        a = sum([int(xi>0) for xi in state[0,7:12]]) 
#                        if a <3:
#                            action = low_act2
#                        else:
                        act_values = alpha*state[0,0:6]+state[0,6:12]
                        action = np.argmax(act_values[1:5])
                            
#                        print(action)
#            dueling_DQN.plot_image(state,kind = 'sucessive')
                    next_raw_state, reward, next_target, vel_state, done = env.step(action)
            next_target = np.reshape(next_target,(1,2)).astype(np.float32)
#            if target[0,0]<=0.2:
#                done = True
            reward = reward+ 0.1*(-next_target[0,0]+target[0,0])
#            print("distance:{},angle:{}".format(next_target[0,0],next_target[0,1]))
#            next_state = tf.image.per_image_standardization(next_state)
            next_raw_state = np.reshape(next_raw_state,(1,100))
            with sess1.as_default():
                with sess1.graph.as_default():
                    next_raw_action1,low_act1 = dueling_DQN1.output(next_target)
            with sess2.as_default():
                with sess2.graph.as_default():
                    next_raw_action2,low_act2 = dueling_DQN2.output(next_raw_state,next_target)
            next_state = np.concatenate((next_raw_action1,next_raw_action2),axis=1)
            next_state = np.reshape(next_state,(1,12))
            print("goal_val: {:.5},stra:{:.4},left1:{:.4},right1:{:.4},left2:{:.4},right2:{:.4}"
                .format(next_state[0,0],next_state[0,1],next_state[0,2],next_state[0,3],next_state[0,4],next_state[0,5]))
            print("obst_val: {:.5},stra:{:.4},left1:{:.4},right1:{:.4},left2:{:.4},right2:{:.4}"
                .format(next_state[0,6],next_state[0,7],next_state[0,8],next_state[0,9],next_state[0,10],next_state[0,11]))
#            next_state = np.append(next_state,state[:,:,:,:1],axis=3)
#            reward = reward if not done else -10
#            next_state = np.reshape(next_state, [1, state_size])
            with sess3.as_default():
                with sess3.graph.as_default():
                    dueling_DQN3.remember(state, action, reward, next_state,done)
                    state = next_state
                    target = next_target
                    total_reward = reward + total_reward
        #            print(action)
        #            if time == 1499:
        #                done = True
                    if done:
                        print("episode: {}/{}, score: {}, e: {:.2}"
                              .format(e, EPISODES, total_reward, dueling_DQN3.epsilon))
                        break
                    if len(dueling_DQN3.memory) == MEMORY_SIZE:
                        dueling_DQN3.learn()
        treward[e] = total_reward
        if len(dueling_DQN3.memory) == MEMORY_SIZE:        
            dueling_DQN3.epsilon_decay_per_epoch()
        if e % 250 == 0:            
            save_path = saver.save(sess3, "/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/HRL/"+str(e)+".ckpt")
            np.save('total_reward.npy',treward)
#        if total_reward > 5:
#            save_path = saver.save(sess3, "/home/zhw/gym-gazebo/examples/turtlebot/dueling_dqn/training/images/HRL/good/"+str(e)+".ckpt")
        if e % 500 == 0:
            save_memory(dueling_DQN3.memory,e,dueling_DQN3.epsilon)
            
            
            
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

