import random
import torch
import numpy as np
import pandas as pd
import sys
from matplotlib import pylab as plt
import itertools
from collections import deque
from sklearn.preprocessing import StandardScaler


class Environment:
    def __init__(self, tm, i_state=0, n_devices=100, lmbda=1.2):

        self.tm = tm
        self.lmbda = lmbda

        self.n_devices = n_devices
        self.states_list = np.array(
            [list(permutacao) for permutacao in itertools.permutations(np.arange(self.n_devices))])
        self.states_list_dict = {tuple(permutacao): indice for indice, permutacao in enumerate(self.states_list)}

        self.len_states_list = len(self.states_list)

        self.actions = [0, 1, 2, 3]
        self.i_state = i_state
        self.state = self.states_list[self.i_state]
        self.v_rand = np.random.rand(1, self.n_devices) / 10.0
        self.scaler = StandardScaler()

    def current_state(self):
        s_ = self.scaler.fit_transform(np.array(self.state).reshape(-1, 1)).reshape(1, -1)[0] + self.v_rand
        return self.i_state, self.state, s_

    def actions_list(self):
        return self.actions

    def len_actions_list(self):
        return len(self.actions)

    def reset(self, i_state):
        self.i_state = i_state
        self.state = self.states_list[self.i_state]
        return self.i_state, self.state

    def act_move(self, action):

        if action == 0:
            self.i_state = (self.i_state - 1) if self.i_state > 0 else self.len_states_list - 1
            self.state = self.states_list[self.i_state]
        elif action == 1:
            self.i_state = (self.i_state + 1) % self.len_states_list
            self.state = self.states_list[self.i_state]
        elif action == 2:
            action_list = list(self.state)
            action_list.insert(0, action_list.pop())
            self.state = tuple(action_list)
            self.i_state = self.states_list_dict.get(self.state, -1)
        else:
            action_list = list(self.state)
            action_list.append(action_list.pop(0))
            self.state = tuple(action_list)
            self.i_state = self.states_list_dict.get(self.state, -1)

    def act(self, action, selected_clients):

        self.act_move(action)

        r = 0
        cont = 0
        for i, rb in enumerate(self.state):
            id_ue = selected_clients[i]

            if self.tm.final_q[id_ue][rb] != 1:
                r = r + self.tm.final_q[id_ue][rb]
                cont = cont + 1

        if cont > 0:
            r = r + self.tm.lmbda * cont * (-1)
        else:
            r = 100

        final = False
        return self.i_state, r, final


class DQN_Opt:
    def __init__(self, transmission_model):
        self.tm = transmission_model
        self.min_fit_clients = self.tm.rb_number
        self.rb_number = self.tm.rb_number

        self.env = Environment(self.tm, 0, self.min_fit_clients)

    def opt(self, selected_clients):

        print("> DQN -> optimization_rb_allocation")
        print(f"initial_clients: {selected_clients.tolist()}")

        ####
        l1 = self.min_fit_clients
        l2 = 150
        l3 = 100
        l4 = self.env.len_actions_list()

        model = torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3, l4)
        )
        loss_fn = torch.nn.MSELoss()
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        ####

        episodes = 300
        steps = 60
        gamma = 0.9

        epsilon = 0.6
        epsilon_decay = 0.99
        epsilon_min = 0.5

        states_visited = []

        mem_size = self.min_fit_clients * self.min_fit_clients
        batch_size = self.min_fit_clients
        replay = deque(maxlen=mem_size)

        epsilon_max = epsilon
        for i in range(episodes):
            self.env.reset(random.randint(0, len(self.env.states_list) - 1))
            epsilon = epsilon_max

            state_i, _, state = self.env.current_state()

            if state_i not in states_visited:
                states_visited.append(state_i)

            state1 = torch.from_numpy(state).float()

            for j in range(steps):

                qval = model(state1)
                qval_ = qval.data.numpy()

                epsilon = max(epsilon * epsilon_decay, epsilon_min)
                if random.random() < epsilon:
                    action_ = random.randint(0, self.env.len_actions_list() - 1)
                else:
                    action_ = np.argmin(qval_)

                newstate, reward, final = self.env.act(action_, selected_clients)

                state_i, _, state2_ = self.env.current_state()
                state2 = torch.from_numpy(state2_).float()

                if state_i not in states_visited:
                    states_visited.append(state_i)

                exp = (state1, action_, reward, state2, final)
                replay.append(exp)

                state1 = state2
                if len(replay) > batch_size:
                    minibatch = random.sample(replay, batch_size)
                    state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                    action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
                    reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
                    state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                    done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])

                    Q1 = model(state1_batch)
                    with torch.no_grad():
                        Q2 = model(state2_batch)

                    Y = reward_batch + gamma * ((1 - done_batch) * torch.min(Q2, dim=1)[0])
                    X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                    loss = loss_fn(X, Y.detach())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # print(f"episode: {i:0{5}}/{episodes}")

        ###################################################
        # Evaluation

        number_of_iterations = len(states_visited)
        max_cont = 0
        min_r = np.inf
        sel_id_ue = 0

        for k in range(5):
            self.env.reset(states_visited[random.randint(0, len(states_visited) - 1)])
            i_state, state_tupla, state = self.env.current_state()

            for i in range(number_of_iterations):

                self.env.reset(i_state)
                i_state, state_tupla, state = self.env.current_state()

                qtde = 0
                r = 0
                for j, item in enumerate(state_tupla):
                    id_ue = selected_clients[item]

                    if self.tm.final_q[id_ue][j] != 1:
                        qtde = qtde + 1
                        r = r + self.tm.final_q[id_ue][j]

                if qtde > 0:
                    r = r + self.tm.lmbda * qtde * (-1)
                else:
                    r = 100

                if qtde > max_cont and r < min_r:
                    max_cont = qtde
                    min_r = r
                    sel_id_ue = i_state

                if random.random() < 0.8:
                    state = torch.from_numpy(state).float()

                    with torch.no_grad():
                        newQ = model(state.reshape(1, self.min_fit_clients))

                    v = newQ.numpy()
                    acao = np.argmin(v[0])

                    self.env.act_move(acao)
                    i_state, _, _ = self.env.current_state()

                else:
                    i_state = states_visited[random.randint(0, len(states_visited) - 1)]

        self.env.reset(sel_id_ue)
        _, state_tupla, _ = self.env.current_state()

        _selected_clients = []
        _rb_allocation = []

        for i, item in enumerate(state_tupla):
            id_ue = selected_clients[item]

            if self.tm.final_q[id_ue][i] < 1:
                _selected_clients.append(id_ue)
                _rb_allocation.append(i)

        return _selected_clients.copy(), _rb_allocation.copy()




