import pulp as pl
import re


class Milp_Opt:
    def __init__(self, transmission_model):

        self.tm = transmission_model
        self.min_fit_clients = self.tm.rb_number
        self.rb_number = self.tm.rb_number

    def opt(self, selected_clients):
        print("> MILP -> optimization_rb_allocation")

        print(f"> optimization_rb_allocation")

        final_number_clients = self.rb_number
        # selected_clients = [2, 13, 16, 23, 24, 27, 28, 30, 56, 83]

        print(f"len(selected_clients): {len(selected_clients)}")
        print(f"rb_number: {self.rb_number}")
        print(f"min_fit_clients: {self.min_fit_clients}")

        print(f"delay_req: {self.tm.delay_requirement}")
        print(f"energy_req: {self.tm.energy_requirement}")
        print(f"selected_clients: {selected_clients}")
        print("*****************")

        # Creation of the assignment problem
        model = pl.LpProblem("Min_Q", pl.LpMinimize)

        # Decision Variables
        x = [[pl.LpVariable(f"x_{i}_{j}", cat=pl.LpBinary) for j in range(self.rb_number)] for i in
             range(self.min_fit_clients)]
        # print(x)

        # Objective function
        model += (
                pl.lpSum(self.tm.q[i][j] * x[i][j] for i in range(self.min_fit_clients) for j in
                         range(self.rb_number)) -
                self.tm.lmbda *
                pl.lpSum(x[i][j] for i in range(self.min_fit_clients) for j in range(self.rb_number))

        ), "Custo_Q_total"

        # Constraints: Each customer is assigned to exactly one channel
        for i in range(self.min_fit_clients):
            model += pl.lpSum(x[i][j] for j in range(self.rb_number)) >= 0, f"Restricao_Cliente_Canal_{i} >= 0"

        for i in range(self.min_fit_clients):
            model += pl.lpSum(x[i][j] for j in range(self.rb_number)) <= 1, f"Restricao_Cliente_Canal_{i} <= 1"

        # Constraints: Each channel is assigned to exactly one customer
        for j in range(self.rb_number):
            model += pl.lpSum(x[i][j] for i in range(self.min_fit_clients)) >= 0, f"Restricao_Canal_Cliente_{j} >= 0"

        for j in range(self.rb_number):
            model += pl.lpSum(x[i][j] for i in range(self.min_fit_clients)) <= 1, f"Restricao_Canal_Cliente_{j} <= 1"

        model += pl.lpSum([x[i][j] for i in range(self.min_fit_clients) for j in
                           range(self.rb_number)]) <= final_number_clients, f"Clientes selecionados"

        for i in range(self.min_fit_clients):
            for j in range(self.rb_number):
                model += x[i][j] * self.tm.user_delay[selected_clients[i]][
                    j] <= self.tm.delay_requirement, f"Restricao_Delay_{i}_{j}"
                model += x[i][j] * self.tm.user_upload_energy[selected_clients[i]][
                    j] <= self.tm.energy_requirement, f"Restricao_Energy_{i}_{j}"
                model += x[i][j] * self.tm.q[selected_clients[i]][
                    j] <= self.tm.error_rate_requirement, f"Restricao_Error_Rate_{i}_{j}"

        ################
        # Resolvendo o problema
        status = model.solve()
        print(pl.LpStatus[status])
        print("Custo total:", pl.value(model.objective))

        _selected_clients = []
        _rb_allocation = []
        for var in model.variables():
            if pl.value(var) == 1:
                # print(var.name)
                indices = [int(i) for i in re.findall(r'\d+', var.name)]
                # print(indices)
                _selected_clients.append(selected_clients[indices[0]])
                _rb_allocation.append(indices[1])

        print("<<<<<<<<<")
        print(_selected_clients)
        return _selected_clients.copy(), _rb_allocation.copy()
