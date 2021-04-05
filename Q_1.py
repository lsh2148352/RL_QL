import numpy as np
import pandas as pd
import time

N_STATES = 6   # 1维世界的宽度
ACTIONS = ['left', 'right']     # 探索者的可用动作
EPSILON = 0.9   # 贪婪度 greedy
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值
MAX_EPISODES = 13   # 最大回合数
FRESH_TIME = 0.01    # 移动间隔时间

def q_table(n_states,actions):#初始化Q表
    z = np.zeros((n_states,len(actions)))
    table = pd.DataFrame(z,columns=actions)
    return table

def choice_action(state,q_table1):#选择动作
    state_action = q_table1.iloc[state,:]#取出这个状态下的所有动作的q值
    print(type(state_action))
    if (np.random.uniform()>EPSILON) or (state_action.all()==0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name =  state_action.argmax()    # 贪婪模式
    return action_name

def get_env_feedback(s,a):#获得环境反馈
    if a == 'right':
        if s == N_STATES - 2:#总的状态是0-5，6步，5就是宝藏，所以N_STATES - 2就是下一步就到宝藏
            s_ = 'terminal'
            r = 1#只有找到宝藏才有奖励
        else:
            s_ = s+1
            r = 0
    else:
        r = 0
        if s == 0:
            s_ = s
        else:
            s_ = s-1
    return s_,r

def update_env(S, episode, step_counter):#更新环境
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    qt = q_table(N_STATES,ACTIONS)
    print(qt)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        s = 0
        is_terminal = False
        update_env(s, episode, step_counter)
        while not is_terminal:
            ac = choice_action(s,qt)
            s_,r = get_env_feedback(s,ac)
            q_pre = qt.loc[s,ac]
            if s_ != 'terminal':
                q_target = r+GAMMA*qt.iloc[s_,:].max()
            else:
                q_target = r
                is_terminal = True

            qt.loc[s,ac] += ALPHA*(q_target-q_pre)
            s = s_
            update_env(s, episode, step_counter+1)  # 环境更新

            step_counter += 1
    return qt

if __name__ == "__main__":
    q= rl()
    print(q)
    