# -*- coding: utf-8 -*-
import sys
import numpy as np
import os
from transmission_model.transmission_model import Transmission_Model
from communication_strategy.communication_strategy import Communication_Strategy
from server.server import Server
from optmizer.milp_optmizer import Milp_Opt
from optmizer.dqn_optmizer import DQN_Opt


class FL(Server):

    def __init__(self, n_rounds, total_number_clients, min_fit_clients, load_client_data_constructor,
                 path_server, path_clients, shape, model_type, optimizer_type, parallel_processing=False):
        super().__init__(n_rounds, total_number_clients, min_fit_clients, load_client_data_constructor,
                         path_server, path_clients, shape, model_type, parallel_processing)

        delay_requirement = 0.2
        energy_requirement = 0.0025
        error_rate_requirement = 0.3
        lmbda = 1.2

        tm = Transmission_Model(rb_number=min_fit_clients, user_number=total_number_clients,
                                total_model_params=self.model.count_params(),
                                delay_requirement=delay_requirement, energy_requirement=energy_requirement,
                                error_rate_requirement=error_rate_requirement, lmbda=lmbda,
                                lower_limit_distance=100, upper_limit_distance=500)

        if optimizer_type == "MILP":
            optmizer = Milp_Opt(tm)
        else:
            optmizer = DQN_Opt(tm)

        self.strategy = Communication_Strategy(
            tm,
            optmizer,
            min_fit_clients=min_fit_clients)
        # delay_requirement=0.2, energy_requirement=0.0025 - NIID R-MNIST com MLP


    def print_result(self):
        print("###############################")
        print(f"centralized_accuracy: ")
        print(self.evaluate_list["centralized"]["accuracy"])

        print(f"centralized_loss: ")
        print(self.evaluate_list["centralized"]["loss"])
        print("###############################")

        for item, value in self.strategy.round_costs_list.items():
            print(item)
            print(value)
            print(np.cumsum(value).tolist())

        print("\ncount_of_client_selected")
        print(self.count_of_client_selected)

        print("\ncount_of_client_uploads")
        print(self.count_of_client_uploads)

    def configure_fit(self):
        # FedAvg
        # self.strategy.random_user_selection(k=self.min_fit_clients)
        # self.strategy.random_rb_allocation()

        # POC
        # self.strategy.greater_loss_user_selection(clients_loss_list=fl.clients_loss, factor=2, k=self.min_fit_clients)
        # self.strategy.random_rb_allocation()

        # FedAvg-Opt
        self.strategy.random_user_selection(k=10)
        self.strategy.optimizer_rb_allocation()

        # POC-Opt
        # self.strategy.greater_loss_user_selection(clients_loss_list=fl.clients_loss, factor=2, k=self.min_fit_clients)
        # self.strategy.optimizer_rb_allocation()

        # sys.exit()
        ################
        self.strategy.upload_status()
        self.strategy.round_costs()
        self.selected_clients = fl.strategy.success_uploads.copy()


if __name__ == "__main__":
    os.system('clear')

    for i in range(1):
        fl = FL(n_rounds=200,
                min_fit_clients=10,
                total_number_clients=100,
                path_server="../datasets/mnist/mnist",
                path_clients="../datasets/mnist/non-iid-0.9-100-rotation-45",
                shape=(28, 28, 1),
                model_type="MLP",
                optimizer_type="DQN",
                load_client_data_constructor=False)

        print(f"INÍCIO")
        evaluate_loss, evaluate_accuracy = None, None

        for fl.server_round in range(fl.n_rounds):

            fl.configure_fit()

            fl.strategy.print_values()
            print(f"success_uploads: {fl.strategy.success_uploads} - error_uploads: {fl.strategy.error_uploads}")
            fl.strategy.print_round_costs()

            for cid in fl.strategy.selected_clients:
                fl.count_of_client_selected[cid] = fl.count_of_client_selected[cid] + 1

            if len(fl.selected_clients) > 0:

                for cid in fl.selected_clients:
                    fl.count_of_client_uploads[cid] = fl.count_of_client_uploads[cid] + 1

                weight_list, sample_sizes, info = fl.fit()

                # Aggregation
                fl.aggregate_fit(weight_list, sample_sizes)

            print(f"***************************")
            # Centralized evaluate
            print(f"Centralized evaluate: R: {fl.server_round + 1} ")
            evaluate_loss, evaluate_accuracy = fl.centralized_evaluation()
            print(f"evaluate_accuracy: {evaluate_accuracy}")
            print(f"***************************")

        fl.print_result()

    print(f"\nFIM -  FL-RL")
