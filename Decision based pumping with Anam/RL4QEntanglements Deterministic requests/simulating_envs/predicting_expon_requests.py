import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gamma, expon
mpl.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Sans'


class ExponPred:
    def __init__(self, request_arrival_rates=[4000,5000], T=10):
    #def __init__(self, R1_arr_rate=0.4, R2_arr_rate=0.6, T=10):
        self.request_arrival_rates = request_arrival_rates
        #self.R1_arr_rate = R1_arr_rate
        #self.R2_arr_rate = R2_arr_rate
        self.T = T
        self.num_particles = 1000
        self.request_process_names_list = []
        arrival_rates_dict = {}
        for i in range(len(self.request_arrival_rates)):
            arrival_rates_dict['R{}_arr_rate'.format(i)] = self.request_arrival_rates[i]
            self.request_process_names_list.append('R_{}'.format(i))
        self.arrival_rates_dict = arrival_rates_dict

        self.prob_distr = []
        prob_of_success_dict = {}
        for i in range(len(self.request_arrival_rates)):
            prob_of_success_dict['prob_of_success_of_R{}'.format(i)] = (
                    arrival_rates_dict['R{}_arr_rate'.format(i)] / sum(self.request_arrival_rates))
            self.prob_distr.append(prob_of_success_dict['prob_of_success_of_R{}'.format(i)])
        self.prob_of_success_dict = prob_of_success_dict
        #self.prob_of_success_of_R1 = R1_arr_rate / (R1_arr_rate+R2_arr_rate)
        #self.prob_of_success_of_R2 = 1 - self.prob_of_success_of_R1
        #self.prob_distr = [self.prob_of_success_of_R1, self.prob_of_success_of_R2]
        self.initialize_particles()
        print()
    def initialize_particles(self):
        # Initialize particles only for Exponential distributions
        self.particles_dict = {}
        self.particles_weights_dict = {}
        for i in range(len(self.request_process_names_list)):
            self.particles_dict['particles_R{}'.format(i)] = np.random.exponential(
                scale=1/self.arrival_rates_dict['R{}_arr_rate'.format(i)], size=self.num_particles)
            self.particles_weights_dict['particles_R{}'.format(i)] = np.ones(self.num_particles) / self.num_particles
        #self.particles_R1 = np.random.exponential(scale=1/self.R1_arr_rate, size=self.num_particles)
        #self.particles_R2 = np.random.exponential(scale=1/self.R2_arr_rate, size=self.num_particles)
        # Initialize weights as uniform for both sets of particles
        #self.weights_R1 = np.ones(self.num_particles) / self.num_particles
        #self.weights_R2 = np.ones(self.num_particles) / self.num_particles

    def update_weights(self, observed_time, request_type):
        if request_type in list(self.request_process_names_list):
            request_type_int = int(request_type.split('_')[1])
            likelihood = (self.particles_dict['particles_R{}'.format(request_type_int)] *
                          np.exp(-self.particles_dict['particles_R{}'.format(request_type_int)] * observed_time))
            self.weights_R1 = likelihood / np.sum(likelihood)
        else:
            raise NotImplementedError

        #if request_type == 'R_1':
         #   likelihoods_R1 = self.particles_R1 * np.exp(-self.particles_R1 * observed_time)
          #  self.weights_R1 = likelihoods_R1 / np.sum(likelihoods_R1)
        #elif request_type == 'R_2':
         #   likelihoods_R2 = self.particles_R2 * np.exp(-self.particles_R2 * observed_time)
          #  self.weights_R2 = likelihoods_R2 / np.sum(likelihoods_R2)
        #else:
         #   raise NotImplementedError

    def weighted_prediction(self, particles, weights):  #todo fix also to use dict
        """Calculate a weighted average or other statistic of predicted times."""
        # This example uses a weighted average, but other approaches could be used
        return np.average(particles, weights=weights)

    def predict_request(self):

        # Use the one with heighest weight
        # next_event_time_R1 = self.particles_R1[np.argmax(self.weights_R1)]
        # next_event_time_R2 = self.particles_R2[np.argmax(self.weights_R2)]

        # Calculate weighted predictions for Exponential distributions
        # next_event_time_R1 = self.weighted_prediction(self.particles_R1, self.weights_R1)
        # next_event_time_R2 = self.weighted_prediction(self.particles_R2, self.weights_R2)

        #choose one particle randomly
        self.next_event_time_dict = {}
        for i in range(len(self.request_process_names_list)):
            self.next_event_time_dict['next_event_time_R{}'.format(i)] = (np.random.choice
                                                                    (self.particles_dict['particles_R{}'.format(i)]))
        #next_event_time_R1 = np.random.choice(self.particles_R1)
        #next_event_time_R2 = np.random.choice(self.particles_R2)

        # Find the earliest next event between Exponential distributions
        next_event_prediction_time = np.random.choice(a=np.arange(len(self.request_arrival_rates)), size=1,
                                                      p=self.prob_distr)[0]
        request_type = 'R_{}'.format(next_event_prediction_time)
        next_event_prediction_time = self.next_event_time_dict['next_event_time_R{}'.format(next_event_prediction_time)]
        #if next_event_prediction_time == next_event_time_R1:
         #   request_type = 'R_1'
        #else:
         #   request_type = 'R_2'

        return next_event_prediction_time, request_type

