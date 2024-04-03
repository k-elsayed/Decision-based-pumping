
import copy
import numpy as np
import json
class benchmark:

    def __init__(self,sample_size=10000, F_R_all=[0], request_arrival_rates=[0], delta_t_g=0.1, T=10,
                 F_E_0=0.7,  eta_db=0.15, l_km=1, p_g=0.8, t_c=1,
                 ):
        self.sample_size=sample_size
        #.R1_arr_rate=R1_arr_rate
        #self.R2_arr_rate = R2_arr_rate
        self.delta_t_g = delta_t_g
        self.T=T
        self.F_E_0=F_E_0
        #self.F_R_1=F_R_1
        #self.F_R_2=F_R_2
        self.eta_db=eta_db
        self.l_km=l_km
        self.p_g=p_g
        self.t_c=t_c
        self.curr_time=0
        self.prev_time=0
        self.req_index=-1
        self.tau_g_event_index=-1
        self.stop_flag = False
        self.F_e=0
        self.req_index=0
        self.flag=0
        self.service=np.empty(0)
        self.request_arrival_rates = request_arrival_rates
        self.F_R_all = F_R_all
        self.F_R_dict = {}
        for i in range(len(self.request_arrival_rates)):
            self.F_R_dict['R_{}'.format(i)] = F_R_all[i]
        self.simulate_sequences_till_time_T()

    def simulate_sequences_till_time_T(self):
        # Use the exponential distribution to generate inter-arrival times
        # Use the exponential distribution to generate inter-arrival times
        self.R_simulated_iat_times_dict = {}
        self.R_event_times_dict = {}
        self.tagged_array_dict = {}
        merged_tagged_request_array = []
        for i in range(len(self.request_arrival_rates)):
            self.R_simulated_iat_times_dict['R{}'.format(i)] = (
                np.random.exponential(scale=1 / self.request_arrival_rates[i], size=self.sample_size))
            event_times = np.cumsum(self.R_simulated_iat_times_dict['R{}'.format(i)])
            event_times = event_times[event_times < self.T]
            self.R_event_times_dict['R{}'.format(i)] = event_times
            self.tagged_array_dict['R{}'.format(i)] = [(num, 'R_{}'.format(i)) for num in
                                                       self.R_event_times_dict['R{}'.format(i)]]
            merged_tagged_request_array += self.tagged_array_dict['R{}'.format(i)]

        # self.R1_simulated_iat_times = np.random.exponential(scale=1 / self.R1_arr_rate, size=self.sample_size)
        # self.R2_simulated_iat_times = np.random.exponential(scale=1 / self.R2_arr_rate, size=self.sample_size)
        # R1_event_times = np.cumsum(self.R1_simulated_iat_times)
        # R2_event_times = np.cumsum(self.R2_simulated_iat_times)
        k_geom_samples = np.random.geometric(self.p_g, size=self.sample_size)
        tau_g_occurances = k_geom_samples * self.delta_t_g
        tau_g_time_series = np.cumsum(tau_g_occurances)

        # R1_event_times = R1_event_times[R1_event_times < self.T]
        # R2_event_times = R2_event_times[R2_event_times < self.T]
        self.tau_g_time_series = tau_g_time_series[tau_g_time_series < self.T]
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

        self.request_times = np.array([item[0] for item in sorted_tagged_request_array])
        self.request_sequence = np.array([item[1] for item in sorted_tagged_request_array])

        self.next_event(0)
        self.prev_time=self.curr_time
    def optimal_decision_event(self):
        self.next_event(0)
        self.reward_v=np.empty(0)
        while not self.stop_flag:
            self.update()  #update the system state based on the self.time
            skip_ent=0
            if (self.flag==0): # flag is 0 if the current event is TE
                # run the sample of the pumpingr
                if(self.F_e==0):
                    self.F_e=self.F_E_0
                else:
                    pump_success_prob = (self.F_e * self.F_E_0) + ((1 - self.F_e) * (1 - self.F_E_0))
                    if pump_success_prob:
                        self.F_e=(self.F_e * self.F_E_0) / pump_success_prob
                    else:
                        self.F_e=max(self.F_e,self.F_E_0)
            else:
                sample_path_fidelity_v, sample_path_waiting_time_v, sample_path_fidelity_teleport_v= self.sample_path_until_next_request()
                skip_ent=self.serve(sample_path_fidelity_teleport_v)

            self.next_event(skip_ent)
        return self.service

    def fidelity_decay (self,time_diff):
        if self.F_e > 0:
            self.F_e = max(0, 0.5 * (
                    1 + (((2 * self.F_e) - 1) * np.exp(-(time_diff / self.t_c)))))  # Eq 1
        else:
            self.F_e = 0

    def fidelity_decay_static (self,fidelity,time_diff):
        if fidelity > 0:
            fidelity = max(0, 0.5 * (
                    1 + (((2 * fidelity) - 1) * np.exp(-(time_diff / self.t_c)))))  # Eq 1
        else:
            fidelity = 0
        return fidelity


    def update (self):
        self.fidelity_decay(self.curr_time-self.prev_time)

    def sample_path_until_next_request(self):
        sample_path_fidelity_v=np.empty(0)
        sample_path_waiting_time_v=np.empty(0)
        sample_path_fidelity_teleport_v=np.empty(0)

        if(self.req_index !=len(self.request_times)-1):
            until_t=self.request_times[self.req_index+1]
        else:
            until_t=self.T

        from_t=self.request_times[self.req_index]
        sample_path_times=[item for item in self.tau_g_time_series if from_t <= item <= until_t]
        current_fidelity=copy.deepcopy(self.F_e)
        waiting_time=0
        sample_path_fidelity_v=np.append(sample_path_fidelity_v,current_fidelity)
        sample_path_waiting_time_v=np.append(sample_path_waiting_time_v,waiting_time)
        sample_path_fidelity_teleport_v = np.append(sample_path_fidelity_teleport_v, current_fidelity * np.exp(-waiting_time / t_c))  ## teleportation fidelity
        for time_sample in sample_path_times:
            current_fidelity=self.fidelity_decay_static(current_fidelity,time_sample-from_t)
            waiting_time+=time_sample-from_t

            if(current_fidelity==0):
                current_fidelity=self.F_E_0
            else:
                pump_success_prob = (current_fidelity * self.F_E_0) + ((1 - current_fidelity) * (1 - self.F_E_0))
                if pump_success_prob:
                    current_fidelity = (current_fidelity * self.F_E_0) / pump_success_prob
                else:
                    current_fidelity = max(current_fidelity, self.F_E_0)
            sample_path_fidelity_v=np.append(sample_path_fidelity_v, current_fidelity)
            sample_path_waiting_time_v=np.append(sample_path_waiting_time_v, waiting_time)
            rewardd=current_fidelity*np.exp(-waiting_time/t_c)
            sample_path_fidelity_teleport_v=np.append(sample_path_fidelity_teleport_v,rewardd)  ## teleportation fidelity

            from_t=time_sample

        return sample_path_fidelity_v, sample_path_waiting_time_v ,sample_path_fidelity_teleport_v

    def serve(self,sample_path_fidelity_teleport_v):

        reward= np.maximum(self.F_r-sample_path_fidelity_teleport_v,0)
        indices = np.where(reward == 0)[0]
        if(len(indices)>0):
            self.service=np.append(self.service,0)
            skip_ent=indices[0]
        else:
            self.service=np.append(self.service,np.amin(reward))
            skip_ent=np.argmin(reward)

        self.flag = 0
        self.waiting_time = 0
        return skip_ent


    def next_event(self,skip_ent):
        #skip the entanglements
        if skip_ent>0:
           # print(self.tau_g_event_index+skip_ent)
            self.curr_time=self.tau_g_time_series[self.tau_g_event_index+skip_ent]
        if(self.curr_time==min(self.request_times[-1],self.tau_g_time_series[-1])): # stopping condition
            self.stop_flag=True
            return
        if (self.req_index == len(self.request_times)-1 or self.tau_g_event_index== len(self.tau_g_time_series)-1):  # stopping condition
            self.stop_flag = True
            return
        time_next_tau_g_event = self.tau_g_time_series[self.tau_g_event_index+1]
        time_next_req = self.request_times[self.req_index + 1]
        time_next_event=min(time_next_tau_g_event,time_next_req)
        self.prev_time=self.curr_time
        self.curr_time=time_next_event

        if time_next_event == time_next_req:
            request_idx = int(self.request_sequence[self.req_index].split('_')[1])
            self.F_r=self.F_R_all[request_idx]
            self.flag=1
            self.req_index+=1
        else:
            self.tau_g_event_index+=1



run_param_path = '/home/karim/My Work/Codes/Decision based pumping with Anam/RL4QEntanglements/Results/Parameter_Files/Eval/run_params_1Pois_eval.json'
run_param = json.load(open(run_param_path, 'r'))
c = run_param['c']
l_km = run_param['l_km']
t_c = run_param['t_c']     # in ms
eta_db = run_param['eta_db']
eta_linear = run_param['eta_linear']
p_g = run_param['p_g']
delta_t = run_param['delta_t']  # time slot
delta_t_g = run_param['delta_t_g']
request_arrival_rates = run_param['request_arrival_rates']
F_E_0 = run_param['F_E_0']
F_R_all= run_param['F_R_all']
sample_size = run_param['sample_size']
T = run_param['T']


Benchmarking=benchmark(sample_size,F_R_all,request_arrival_rates, delta_t_g, T,
                 F_E_0,  eta_db, l_km, p_g, t_c)

reward_v=Benchmarking.optimal_decision_event()
print(np.mean(reward_v))
print(len(reward_v))