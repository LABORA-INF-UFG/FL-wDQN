import numpy as np


class Communication_Strategy:

    def __init__(self, transmission_model, optmizer, min_fit_clients):

        self.tm = transmission_model
        self.optmizer = optmizer
        self.min_fit_clients = min_fit_clients


        self.count_selected_clients = 0
        self.selected_clients = np.array([])
        self.rb_allocation = np.array([])

        self.success_uploads = []
        self.error_uploads = []

        self.round_costs_list = {
            'total_training': [],
            'total_uploads': [],
            'total_error_uploads': [],
            'energy_success': [],
            'energy_error': [],
            'total_energy': [],
            'delay': [],
            'bw': [],
            'power': []
        }


    def greater_loss_user_selection(self, clients_loss_list, factor, k):
        selected_clients = np.random.permutation(self.tm.user_number)[:int(self.min_fit_clients * factor)]
        print(f"user_selection: {selected_clients}")

        loss_samples_list = np.array(clients_loss_list)[selected_clients]
        pos_list = np.arange(len(loss_samples_list))
        print(loss_samples_list)

        # Combination of lists
        combined_data = list(zip(loss_samples_list, pos_list))
        sorted_data = sorted(combined_data, reverse=True, key=lambda x: x[0])
        loss_list, pos_list = zip(*sorted_data)

        print(f"data_list: {loss_list}")
        print(f"pos_list: {pos_list}")

        final_selected_clients = np.sort(selected_clients[np.array(pos_list)[:k]])
        print(f"final user_selection: {final_selected_clients}")

        self.selected_clients = final_selected_clients
        self.count_selected_clients = len(self.selected_clients)

    def random_user_selection(self, k):
        self.selected_clients = np.zeros(self.tm.user_number, dtype=int)
        self.selected_clients[np.random.permutation(self.tm.user_number)[:k]] = 1
        self.selected_clients = np.where(self.selected_clients > 0)[0]
        self.count_selected_clients = len(self.selected_clients)

    def random_rb_allocation(self):
        self.rb_allocation = np.zeros(self.tm.user_number, dtype=int)
        self.rb_allocation[np.random.permutation(self.tm.rb_number)[:self.min_fit_clients]] = 1
        self.rb_allocation = np.random.permutation(np.where(self.rb_allocation > 0)[0])

    def print_values(self):
        print("----------------------------------")
        print(f"selected_clients: {self.selected_clients}")
        if len(self.selected_clients) > 0:
            print(f"rb_allocation: {self.rb_allocation}")

        print("----------------------------------")

    def upload_status(self):
        prob = np.random.rand(self.count_selected_clients)

        self.success_uploads = []
        self.error_uploads = []

        for i, ue in enumerate(self.selected_clients):
            prob_w = self.tm.W[ue, self.rb_allocation[i]]
            q_f = self.tm.final_q[ue, self.rb_allocation[i]]
            print(
                f"{i} - {ue} --> q: {q_f:.4f} W: {prob_w:.4f} - P: {prob[i]:.4f} {'' if prob_w > 0 and prob_w >= prob[i] else ' - [X]'}")

            if prob_w > 0 and prob_w >= prob[i]:
                self.success_uploads.append(ue)
            else:
                self.error_uploads.append(ue)

    def round_costs(self):

        total_training = len(self.selected_clients)
        total_uploads = len(self.success_uploads)

        round_energy = 0
        round_energy_success = 0
        round_energy_error = 0
        round_delay = 0
        round_bw = 0
        round_power = 0

        for i, ue in enumerate(self.selected_clients):
            # print(ue)
            round_bw = round_bw + self.tm.user_bandwidth
            round_power = round_power + self.tm.user_power
            round_energy = round_energy + self.tm.total_energy[ue, self.rb_allocation[i]]
            round_delay = round_delay + self.tm.total_delay[ue, self.rb_allocation[i]]

            # print(f"---*> {ue} - {self.success_uploads} - {self.selected_clients}")
            if ue in self.success_uploads:
                round_energy_success = round_energy_success + self.tm.total_energy[ue,  self.rb_allocation[i]]
            else:
                round_energy_error = round_energy_error + self.tm.total_energy[ue, self.rb_allocation[i]]

        self.round_costs_list['total_training'].append(total_training)
        self.round_costs_list['total_uploads'].append(total_uploads)
        self.round_costs_list['total_error_uploads'].append(total_training - total_uploads)
        self.round_costs_list['energy_success'].append(round_energy_success)
        self.round_costs_list['energy_error'].append(round_energy_error)
        self.round_costs_list['total_energy'].append(round_energy)
        self.round_costs_list['delay'].append(round_delay)
        self.round_costs_list['bw'].append(round_bw)
        self.round_costs_list['power'].append(round_power)

    def print_round_costs(self):
        print("------------------------------------")
        print(f"total_training: {self.round_costs_list['total_training'][-1]}")
        print(
            f"total_uploads: {self.round_costs_list['total_uploads'][-1]}/{self.round_costs_list['total_training'][-1]}")
        print(
            f"total_error_uploads: {(self.round_costs_list['total_training'][-1] - self.round_costs_list['total_uploads'][-1])}/{self.round_costs_list['total_training'][-1]}")
        print("------------------------------------")

    def optimizer_rb_allocation(self):
        self.selected_clients, self.rb_allocation = self.optmizer.opt(self.selected_clients)

