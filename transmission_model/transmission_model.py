from typing import Any
import numpy as np


class Transmission_Model:

    def __init__(self, rb_number, user_number, total_model_params, delay_requirement, energy_requirement,
                 error_rate_requirement, lmbda=1.2, lower_limit_distance=100, upper_limit_distance=500):
        self.rb_number = rb_number
        self.user_number = user_number
        self.total_model_params = total_model_params
        self.delay_requirement = delay_requirement
        self.energy_requirement = energy_requirement
        self.error_rate_requirement = error_rate_requirement
        self.lmbda = lmbda

        self.lower_limit_distance = lower_limit_distance
        self.upper_limit_distance = upper_limit_distance

        self.data_size_model = 0
        self.N = 10 ** -20
        self.q = np.array([])
        self.h = np.array([])

        self.user_power = 0.01  # W -> 10dBm
        self.user_bandwidth = 1  # MHz
        self.user_interference = np.array([])
        self.user_distance = np.array([])
        self.user_angles = np.array([])
        #
        self.user_sinr = np.array([])
        self.user_data_rate = np.array([])
        self.user_delay = np.array([])

        self.base_station_power = 1  # W -> 30dBm
        self.base_station_bandwidth = 20  # MHz
        #
        self.base_station_sinr = np.array([])
        self.base_station_data_rate = np.array([])
        self.base_station_delay = np.array([])

        self.total_delay = np.array([])

        self.energy_coeff = 10 ** (-27)
        self.cpu_cycles = 40
        self.cpu_freq = 10 ** 9
        self.user_energy_training = np.array([])
        self.user_upload_energy = np.array([])
        self.total_energy = np.array([])

        self.W = np.array([])
        self.final_q = np.array([])

        print(f">>>>>>>>>>>>> total_model_params: {total_model_params}")

        self.init()

    def init(self):
        self.init_user_interference()
        self.init_distance()
        self.init_q()  #
        self.init_h()  #
        self.init_user_sinr()  #
        self.init_user_data_rate()  #
        self.init_base_station_sinr()
        self.init_base_station_data_rate()
        self.init_data_size_model()
        self.init_user_delay()
        self.init_base_station_delay()
        self.init_totaldelay()
        self.init_user_energy_training()
        self.init_user_upload_energy()
        self.init_total_energy()
        self.compute_transmission_probability_matrix()
        self.compute_final_q()


    def init_user_interference(self):
        i = np.array([0.05 + i * 0.01 for i in range(self.rb_number)])
        self.user_interference = (i - 0.04) * 0.000001

    def init_distance(self):
        np.random.seed(1)
        self.user_distance, self.user_angles = self.lower_limit_distance + (
                self.upper_limit_distance - self.lower_limit_distance) * np.random.rand(self.user_number,
                                                                                        1), 2 * np.pi * np.random.rand(
            self.user_number)
        np.random.seed()

    def init_q(self):
        # Packet error rate of each user over each RB
        self.q = 1 - np.exp(-1.08 * (self.user_interference + self.N * self.user_bandwidth) /
                            (self.user_power * (self.user_distance ** -2)))

    def init_h(self):
        # Rayleigh fading parameter
        o = 1
        self.h = o * (self.user_distance ** (-2))

    def init_user_sinr(self):
        self.user_sinr = self.user_power * self.h / (self.user_interference + self.user_bandwidth * self.N)

    def init_user_data_rate(self):
        self.user_data_rate = self.user_bandwidth * np.log2(1 + self.user_sinr)
        """
        The frequency of 1 MHz means 1 million symbols per second (1 MSps or 1 Mbaund).
        The relationship between baud and bits per second (bps) depends on the modulation technique and the number of 
        bits represented by each symbol.        
        Considering a technique, which modulates 8 bits for each symbol, the bit rate per symbol is equivalent to 1 byte.
        Therefore, with the frequency of 1 MHz, the data rate expressed in Mbaunds is equivalent to MB/s.
        """

    def init_base_station_sinr(self):
        base_station_interference = 0.06 * 0.000003  # Interference over downlink
        self.base_station_sinr = (self.base_station_power * self.h /
                                  (base_station_interference + self.N * self.base_station_power))

    def init_base_station_data_rate(self):
        self.base_station_data_rate = self.base_station_bandwidth * np.log2(1 + self.base_station_sinr)

    def init_data_size_model(self):
        # MBytes
        self.data_size_model = self.total_model_params * 4 / (1024 ** 2)

    def init_user_delay(self):
        self.user_delay = self.data_size_model / self.user_data_rate

    def init_base_station_delay(self):
        self.base_station_delay = self.data_size_model / self.base_station_data_rate

    def init_totaldelay(self):
        self.total_delay = self.user_delay + self.base_station_delay

    def init_user_energy_training(self):
        self.user_energy_training = self.energy_coeff * self.cpu_cycles * (self.cpu_freq ** 2) * self.data_size_model

    def init_user_upload_energy(self):
        self.user_upload_energy = self.user_power * self.user_delay

    def init_total_energy(self):
        self.total_energy = self.user_energy_training + self.user_upload_energy

    def compute_transmission_probability_matrix(self):
        self.W = np.zeros((self.user_number, self.rb_number))
        for i in range(self.user_number):
            for j in range(self.rb_number):
                if self.user_delay[i, j] <= self.delay_requirement and self.user_upload_energy[i, j] <= self.energy_requirement and self.q[i, j] <= self.error_rate_requirement:
                    self.W[i, j] = 1 - self.q[i, j]

    def compute_final_q(self):
        self.final_q = np.ones((self.user_number, self.rb_number))
        for i in range(self.user_number):
            for j in range(self.rb_number):
                if self.user_delay[i, j] <= self.delay_requirement and self.user_upload_energy[i, j] <= self.energy_requirement and self.q[i, j] <= self.error_rate_requirement:
                    self.final_q[i, j] = self.q[i, j]

