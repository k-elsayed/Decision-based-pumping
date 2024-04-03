import scipy.stats as st
import gym
# import torch
from gym.spaces import Discrete, Box, Tuple
import numpy as np
# import torch.nn.functional as f
import torch

class Q_env(gym.Env):
    # def __init__(self, sample_size=1000, request_arrival_rates=np.array([4000,5000]), delta_t_g=0.1, T=10,
    #              F_E_0=0.7, F_R_1=0.5, F_R_2=0.5, pred_env=None, eta_db=0.15, eta_linear=0.1, l_km=1, p_g=0.8, t_c=1,
    #              ppo_actions_discrete=False, trained_policy=''):
    def __init__(self, sample_size=1000, request_arrival_rates=np.array([4000, 5000]), delta_t_g=0.1, T=10,
                 F_E_0=0.7, F_R_all=np.array([0.5,0.5]), pred_env=None, eta_db=0.15, eta_linear=0.1, l_km=1, p_g=0.8,
                 t_c=1, ppo_actions_discrete=False, trained_policy='', eval=True):
        self.eval = eval
        # self.device = 'cpu'
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ppo_actions_discrete = ppo_actions_discrete
        # self.event_time_series = event_time_series
        # self.event_sequence = event_sequence
        # self.request_sequence = request_sequence
        # self.request_times = request_times
        # self.tau_g_time_series = tau_g_time_series
        # self.R1_simulated_iat_times = R1_simulated_iat_times
        # self.R2_simulated_iat_times = R2_simulated_iat_times
        self.policy = trained_policy
        self.request_arrival_rates = request_arrival_rates
        #self.R1_arr_rate = R1_arr_rate
        #self.R2_arr_rate = R2_arr_rate
        self.delta_t_g = delta_t_g
        self.T = T
        self.max_episode_steps = T/5
        self.event_samples=sample_size
        self.p_g = p_g
        self.F_E_0 = F_E_0
        # self.F_R_1 = F_R_1
        # self.F_R_2 = F_R_2
        self.F_R_all = F_R_all
        self.F_R_dict = {}
        for i in range(len(self.request_arrival_rates)):
            self.F_R_dict['R_{}'.format(i)] = F_R_all[i]
        self.pred_env = pred_env
        self.sample_size = sample_size
        #self.max_waiting_time = T
        self.max_waiting_time = - np.log(0.5) * t_c
        self.t_c = t_c
        self.eta_db=eta_db
        self.eta_linear=eta_linear
        self.l_km=l_km
        # state is [F_r, flag, F_e]
        # self.observation_space = Tuple((Box(low=0, high=1, shape=(1,), dtype=np.float32),
        #                                Discrete(2),
        #                                Box(low=0, high=1, shape=(1,), dtype=np.float32)))
        self.observation_space = Box(low=np.array([0, 0, 0, 0]), high=np.array([self.max_waiting_time, 1, 1, 1]), shape=(4,), dtype=np.float32)
        # self.observation_space = Tuple((Box(low=0, high=1, shape=(3,), dtype=np.float32),
        #                                 Box(low=0, high=self.max_waiting_time, shape=(1,), dtype=np.float32)))

        if self.ppo_actions_discrete:
            # discrete action for [serve, not serve], [pump, not pump]
            self.action_space = Tuple((Discrete(2), Discrete(2)))
        else:
            # action probabilities for [serve, not serve], [pump, not pump]
            self.action_space = Tuple((Box(low=0, high=1, shape=(2,), dtype=np.float32),
                                      Box(low=0, high=1, shape=(2,), dtype=np.float32)))


    def simulate_sequences_till_time_T(self):       # todo change to number of event T instead of time
        # Use the exponential distribution to generate inter-arrival times
        self.R_simulated_iat_times_dict = {}
        self.R_event_times_dict = {}
        self.tagged_array_dict = {}
        merged_tagged_request_array = []
        for i in range(len(self.request_arrival_rates)):
            self.R_simulated_iat_times_dict['R{}'.format(i)] = (
                np.random.exponential(scale=1 / self.request_arrival_rates[i], size=self.sample_size))
            event_times = np.cumsum(self.R_simulated_iat_times_dict['R{}'.format(i)])
            #event_times = event_times[event_times < self.T]
            self.R_event_times_dict['R{}'.format(i)] = event_times
            self.tagged_array_dict['R{}'.format(i)] = [(num, 'R_{}'.format(i)) for num in self.R_event_times_dict['R{}'.format(i)]]
            merged_tagged_request_array += self.tagged_array_dict['R{}'.format(i)]

        #self.R1_simulated_iat_times = np.random.exponential(scale=1 / self.R1_arr_rate, size=self.sample_size)
        #self.R2_simulated_iat_times = np.random.exponential(scale=1 / self.R2_arr_rate, size=self.sample_size)
        #R1_event_times = np.cumsum(self.R1_simulated_iat_times)
        #R2_event_times = np.cumsum(self.R2_simulated_iat_times)
        k_geom_samples = np.random.geometric(self.p_g, size=self.sample_size)
        tau_g_occurances = k_geom_samples * self.delta_t_g
        tau_g_time_series = np.cumsum(tau_g_occurances)

        #R1_event_times = R1_event_times[R1_event_times < self.T]
        #R2_event_times = R2_event_times[R2_event_times < self.T]
        #self.tau_g_time_series = tau_g_time_series[tau_g_time_series < self.T]
        self.tau_g_time_series = tau_g_time_series

        self.tagged_array_dict['TE'] = [(num, 'TE') for num in self.tau_g_time_series]


        # Tag each element in the arrays with an identifier for its original array
        # tagged_array1 = [(num, 'R_1') for num in R1_event_times]
        # tagged_array2 = [(num, 'R_2') for num in R2_event_times]
        # tagged_array3 = [(num, 'TE') for num in self.tau_g_time_series]


        # Merge the tagged arrays
        merged_tagged_array = merged_tagged_request_array + self.tagged_array_dict['TE']

        # merged_tagged_array = tagged_array1 + tagged_array2 + tagged_array3
        # merged_tagged_request_array = tagged_array1 + tagged_array2

        # Sort the merged array by the numbers
        sorted_tagged_array = sorted(merged_tagged_array, key=lambda x: x[0])
        sorted_tagged_request_array = sorted(merged_tagged_request_array, key=lambda x: x[0])

        # Split the sorted, tagged array into two arrays: numbers and tags
        self.event_time_series = np.array([item[0] for item in sorted_tagged_array])
        self.event_sequence = np.array([item[1] for item in sorted_tagged_array])
        #print("event_length",len(self.event_sequence))

        self.request_times = np.array([item[0] for item in sorted_tagged_request_array])
        self.request_sequence = np.array([item[1] for item in sorted_tagged_request_array])
        print("")

    def reset(self):
        # todo reset the times
        self.event_counter=0
        self.simulate_sequences_till_time_T()
        self.F_e = self.F_E_0
        self.F_actual_r_time = 0
        self.curr_t = 0
        self.prev_t = 0
        self.real_request_event_index = 0
        # self.R1_event_index = 0
        # self.R2_event_index = 0
        self.R_event_index = {}
        for i in range(len(self.request_arrival_rates)):
            self.R_event_index['R{}'.format(i)] = 0
        self.time_series_event_index = 0
        self.tau_g_event_index = 0
        self.all_pred_request_times = []
        self.all_pred_request_types = []
        F_pred_r_iat, self.pred_event_type = self.pred_env.predict_request()
        self.F_pred_r_time = self.curr_t + F_pred_r_iat
        self.all_pred_request_types.append(self.pred_event_type)
        self.all_pred_request_times.append(self.F_pred_r_time)

        # if self.pred_event_type == 'R_1':
        #     self.F_pred_r = self.F_R_1
        # else:
        #     self.F_pred_r  = self.F_R_2
        self.F_pred_r  = self.F_R_all[int(self.pred_event_type.split('_')[1])]
        self.flag = 0
        self.curr_obs = np.array([0, self.F_pred_r, self.flag, self.F_e], dtype=np.float32)
        # self.curr_obs = (np.array([self.F_pred_r], dtype=np.float32), self.flag,
        #                  np.array([self.F_e], dtype=np.float32))
        self.info = {}
        self.done = False
        self.all_events = ['T_E']
        self.request_queue = []
        self.waiting_time_request_queue = []
        self.pumping = True
        return self.curr_obs

    def step(self, action):
        # actions only at tau_g or real_request times not at predictions
        if self.policy in ('QE_Pois', 'QE_Pois_PPO'):
            serve_action, pump_action = self.normalise_actions(action)    # todo: add action masking later
        else:
            serve_action = np.array(action[0])
            pump_action = np.array(action[1])

        reward = self.simulate_until_next_event_time(serve_action, pump_action)
        self.simulate_next_event()

        self.prev_t = self.curr_t
        self.curr_t = self.time_actual_occuring_event
        self.time_passed = self.curr_t - self.prev_t
        if self.flag == 0:
            self.curr_obs = np.array([0, self.F_pred_r, self.flag, self.F_e], dtype=np.float32)
            # self.curr_obs = (np.array([self.F_pred_r], dtype=np.float32), self.flag,
            #                  np.array([self.F_e], dtype=np.float32))
        else:
            self.curr_obs = np.array([self.waiting_time_request_queue[0], self.F_r, self.flag, self.F_e],
                                     dtype=np.float32)
            # self.curr_obs = (np.array([self.F_r], dtype=np.float32), self.flag,
            #          np.array([self.F_e], dtype=np.float32))

        if self.event_counter >= self.event_samples:
            self.info["observation"] = self.curr_obs
            self.done = True
        # print(self.curr_obs, reward, self.done, self.info)
        # if self.curr_t >= self.max_episode_steps:
        self.event_counter += 1
        return self.curr_obs, reward, self.done, self.info

    def fidelity_decay (self,time_diff):
        if self.F_e > 0:
            self.F_e = max(0, 0.5 * (
                    1 + (((2 * self.F_e) - 1) * np.exp(-(time_diff / self.t_c)))))  # Eq 1
        else:
            self.F_e = 0

    def simulate_until_next_event_time(self, serve_action, pump_action):
        reward_pump = 0
        reward_serve = 0
        curr_F_e_curr_t = 0
        add_reward_flag=False
        # fidelity decay for the time between the prev event and this event
        self.fidelity_decay(self.curr_t - self.prev_t)

        ## Changes based on the action
        if self.all_events[-1] == 'T_E':
            if self.F_e > 0:
                pump = np.random.choice(a=[0, 1], p=pump_action, size=1)[0]
                # check if pumping successful and calculate the new fidelity at this time
                if pump:
                    # pump and update fidelity
                    # success prob denominator eq 2 bernoulli
                    pump_success_prob = (self.F_e * self.F_E_0) + ((1 - self.F_e)*(1-self.F_E_0))
                    success_pump = st.bernoulli.rvs(p=pump_success_prob)
                    if success_pump:
                        curr_F_e_curr_t = (self.F_e * self.F_E_0) / pump_success_prob       # Eq 2
                    else:
                        curr_F_e_curr_t = 0     # unsuccessful pumping
                else:
                    curr_F_e_curr_t=max(self.F_E_0,self.F_e) ##
            else:
                # if no existing entanglement then a new one is created at event 'T_E' with fidelity 0.75
                curr_F_e_curr_t = self.F_E_0
            if not self.eval:       # to be used only for PPO training not for evaluating or other baseline algos
                reward_pump = self.reward_pump(pump_action, curr_F_e_curr_t)

        if len(self.request_queue) > 0:
            waiting_time = 0
            # todo add timeout of requests
            self.serve = np.random.choice(a=[0, 1], p=serve_action, size=1)[0]
            # update flag to 0 if served successfully, else keep it 1
            if self.serve:
                if(self.eval):
                    add_reward_flag=True
                curr_F_e_curr_t = 0
                self.request_queue.pop(0)
                waiting_time = self.waiting_time_request_queue.pop(0)
                if len(self.request_queue) == 0:
                    self.flag = 0

            reward_serve = self.reward_serve(serve_action, pump_action, curr_F_e_curr_t, waiting_time)

        if len(self.request_queue) > 0:     # update waiting time to all still in queue
            self.waiting_time_request_queue += (self.curr_t - self.prev_t)
            self.waiting_time_request_queue = self.waiting_time_request_queue.tolist()

        reward = reward_pump + reward_serve
        #if reward > 0 or reward < -1:
           # print("reward",reward)
        self.F_e=curr_F_e_curr_t
        if(not self.eval):
            return reward
        else:
            if(add_reward_flag):
                if(self.F_r==0.85):
                    print("")
                return reward
            else:
                return np.NAN

    def reward_pump(self, action, curr_F_e_curr_t):
        if self.flag:
            #reward = -(self.F_r - curr_F_e_curr_t)
            reward=- max(self.F_r - self.teleportation_fd(curr_F_e_curr_t,self.waiting_time_request_queue[-1]),0)

        else:
            reward= -max(self.F_pred_r - self.teleportation_fd(curr_F_e_curr_t,0),0)
            #reward = -(self.F_pred_r - curr_F_e_curr_t)
            # or F_pred_r- F_e(next_pred_time) --------------
        return reward

    def reward_serve(self, serve_action, pump_action, curr_F_e_curr_t, waiting_time):
        old_F_e = self.F_e
        reward = 0
        if self.serve:
            #reward += (np.exp(-(waiting_time)))
            #reward -= (self.F_r - old_F_e)
            reward=-max(self.F_r - self.teleportation_fd(self.F_e, waiting_time), 0)

        else:
            #reward -= (np.exp(-(self.waiting_time_request_queue[0])))
            expected_waiting_time_pois = self.waiting_time_request_queue[0]+self.delta_t_g  # added by KE
            #reward += (np.exp(-expected_waiting_time_pois)) # added by KE
            if(self.F_e>0):
                nextfidelity = max(0, 0.5 * ( 1 + (((2 * self.F_e) - 1) * np.exp(-(self.delta_t_g / self.t_c)))))  # Eq 1
                expected_pump_success=(nextfidelity * self.F_E_0) + ((1 - nextfidelity)*(1-self.F_E_0))
                next_pumped_fidelity=(nextfidelity*self.F_E_0)/expected_pump_success

                #expected_fidelity=pump_action[1]*(expected_pump_success*next_pumped_fidelity)+pump_action[0]*max(self.F_E_0,next_pumped_fidelity) # ----- KE: assumming pump_action[1] means to pump

                expected_fidelity=(expected_pump_success*next_pumped_fidelity) # ----- KE: assumming we have to pump if we decide to wait
            else:
                expected_fidelity=self.F_E_0
            #reward -= (self.F_r - expected_fidelity)
            #print("expected", expected_fidelity)
            reward =- max(self.F_r - self.teleportation_fd(expected_fidelity,expected_waiting_time_pois),0)


        return reward
    def teleportation_fd(self,fidelity,waiting_time):
        return fidelity * np.exp(-waiting_time)
    def normalise_actions(self, action):
        if self.ppo_actions_discrete:
            pump_action = np.zeros(2)
            serve_action = np.zeros(2)
            sa = action[0]
            pa = action[1]
            pump_action[pa] = 1
            serve_action[sa] = 1
        else:
            sa = action[0] + 1e-10
            serve_action = sa / sa.sum()
            pa = action[1] + 1e-10
            pump_action = pa / pa.sum()
        return serve_action, pump_action

    def simulate_next_event(self):
        if len(self.request_queue) > 0 and self.serve == 1:
            next_R = self.request_queue[0]
            # if next_R == 'R_1':
            #     self.F_r = self.F_R_1
            # else:
            #     self.F_r = self.F_R_2
            self.F_pred_r = self.F_R_all[int(next_R.split('_')[1])]
            self.flag = 1  # ---------- KE: why?


        # simulate next event times
        self.time_next_actual_request_event = self.request_times[self.real_request_event_index]
        self.time_next_tau_g_event = self.tau_g_time_series[self.tau_g_event_index]
        self.next_actual_event_type = self.request_sequence[self.real_request_event_index]
        request_idx = int(self.next_actual_event_type.split('_')[1])
        self.time_actual_occuring_event = min(self.time_next_tau_g_event, self.time_next_actual_request_event)

        if self.time_actual_occuring_event == self.time_next_tau_g_event:       #next event is TE
            F_pred_r_iat, pred_event_type = self.pred_env.predict_request()
            F_pred_r_time = self.curr_t + F_pred_r_iat
            prev_pred_time = F_pred_r_time
            prev_pred_event = pred_event_type
            prev_F_pred_r_iat = F_pred_r_iat
            # keep simulating predictions until the next TE occurs  ---- KE, can be done only at the events
            while F_pred_r_time < self.time_next_tau_g_event:
                prev_pred_time = F_pred_r_time
                prev_pred_event = pred_event_type
                prev_F_pred_r_iat = F_pred_r_iat
                F_pred_r_iat, pred_event_type = self.pred_env.predict_request()
                self.all_pred_request_times.append(F_pred_r_time)
                self.all_pred_request_types.append(pred_event_type)
                F_pred_r_time += F_pred_r_iat
            # use the last prediction for self.curr_obs
            self.F_pred_r_time = prev_pred_time
            self.F_pred_r_iat = prev_F_pred_r_iat
            self.pred_event_type = prev_pred_event
            # if self.pred_event_type == 'R_1':
            #     self.F_pred_r = self.F_R_1
            # else:
            #     self.F_pred_r = self.F_R_2
            self.F_pred_r = self.F_R_all[int(self.pred_event_type.split('_')[1])]
            self.tau_g_event_index += 1
            self.all_events.append('T_E')

        else:
            # if self.next_actual_event_type == 'R_1':        # next event is actual request
            if len(self.request_queue) == 0:
                # self.F_r = self.F_R_1
                self.F_r = self.F_R_all[request_idx]
            iat_time = self.R_simulated_iat_times_dict['R{}'.format(request_idx)][self.R_event_index['R{}'.format(request_idx)]]
            self.R_event_index['R{}'.format(request_idx)] += 1
            # iat_time = self.R1_simulated_iat_times[self.R1_event_index]
            # self.R1_event_index += 1
            # else:
            #     if len(self.request_queue) == 0:
            #         self.F_r = self.F_R_2
            #     iat_time = self.R2_simulated_iat_times[self.R2_event_index]
            #     self.R2_event_index += 1
            self.flag = 1

            # if real request comes then update weights
            self.pred_env.update_weights(iat_time, self.next_actual_event_type)
            self.real_request_event_index += 1

            self.all_events.append(self.next_actual_event_type)
            self.request_queue.append(self.next_actual_event_type)
            self.waiting_time_request_queue.append(0)





