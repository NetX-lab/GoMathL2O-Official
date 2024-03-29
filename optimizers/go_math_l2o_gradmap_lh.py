# Modified from lstm2. new models: p,a,b,b1,b2

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as Func
import torch.nn.init as init

from collections import defaultdict
from optimizees.base import BaseOptimizee
SQRT2 = np.sqrt(2.0)

NORM_FUNC = {
    'exp': torch.exp,
    'eye': nn.Identity(),
    'sigmoid4': lambda x: 4.0 * torch.sigmoid(x),
    'sigmoid3': lambda x: 3.0 * torch.sigmoid(x),
    'sigmoid': lambda x: 2.0 * torch.sigmoid(x),
    'sigmoid1': lambda x: torch.sigmoid(x),
    'sigmoid05': lambda x: SQRT2 * torch.sigmoid(x),
    'tanh': lambda x: 2 * torch.tanh(x),
    'softplus': nn.Softplus(),
    'Gaussian': lambda x: 2.0 - 2.0*torch.exp(-x**2),
    'Gaussian1': lambda x: 1.0 - 1.0*torch.exp(-x**2),
}


class GOMathL2OLH(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, layers,
                 r_use=True, r_scale=1.0, r_scale_learned=True, r_norm='eye',
                 q_use=True, q_scale=1.0, q_scale_learned=True, q_norm='eye',
                 b_use=True, b_scale=1.0, b_scale_learned=True, b_norm='eye',
                 b3_use=True, b3_scale=1.0, b3_scale_learned=True, b3_norm='eye',
                 b4_use=True, b4_scale=1.0, b4_scale_learned=True, b4_norm='eye',
                 **kwargs):
        """
        Coordinate-wise non-smooth version of our proposed model.
                Please check (18) and (19) in the following paper:
                Liu et al. (2023) "Towards Constituting Mathematical Structures for Learning to Optimize."

                NOTE: r is p in draft.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        use_bias = True

        self.hist = defaultdict(list)

        self.layers = layers  # Number of layers for LSTM

        # self.linear_lstm = nn.Linear(input_size, hidden_size, bias=use_bias)

        self.lstm = nn.LSTM(input_size, hidden_size, layers, bias=use_bias)
        # one more hidden laer before the output layer.
        # borrowed from NA-ALISTA: https://github.com/feeds/na-alista
        self.linear = nn.Linear(hidden_size, hidden_size, bias=use_bias)

        # pre-conditioner
        self.linear_r = nn.Linear(hidden_size, output_size, bias=use_bias)
        self.linear_q = nn.Linear(hidden_size, output_size, bias=use_bias)

        # bias
        self.linear_b3 = nn.Linear(hidden_size, output_size, bias=use_bias)
        self.linear_b4 = nn.Linear(hidden_size, output_size, bias=use_bias)

        # momentum
        self.linear_b = nn.Linear(hidden_size, output_size, bias=use_bias)

        self.state = None
        self.step_size = kwargs.get('step_size', None)
        self.B_step_size = None
        self.C_step_size = None

        self.r_use = r_use
        self.r_scale = nn.Parameter(torch.tensor(
            1.) * r_scale) if r_scale_learned else r_scale
        self.r_norm = NORM_FUNC[r_norm]

        self.q_use = q_use
        self.q_scale = nn.Parameter(torch.tensor(
            1.) * q_scale) if q_scale_learned else q_scale
        self.q_norm = NORM_FUNC[q_norm]

        # Momentum

        self.b_use = b_use
        self.b_scale = nn.Parameter(torch.tensor(
            1.) * b_scale) if b_scale_learned else b_scale
        self.b_norm = NORM_FUNC[b_norm]

        self.b3_use = b3_use
        if b3_scale_learned:
            self.b3_scale = nn.Parameter(torch.tensor(1.) * b3_scale)
        else:
            self.b3_scale = b3_scale
        self.b3_norm = NORM_FUNC[b3_norm]

        self.b4_use = b4_use
        if b4_scale_learned:
            self.b4_scale = nn.Parameter(torch.tensor(1.) * b3_scale)
        else:
            self.b4_scale = b4_scale
        self.b4_norm = NORM_FUNC[b4_norm]

        self._init_parameters()

    def _init_parameters(self):
        init.xavier_uniform_(self.linear.weight)

        init.xavier_uniform_(self.linear_r.weight)
        init.xavier_uniform_(self.linear_q.weight)

        init.xavier_uniform_(self.linear_b.weight)

        init.xavier_uniform_(self.linear_b3.weight)
        init.xavier_uniform_(self.linear_b4.weight)

        init.zeros_(self.linear.bias)
        init.zeros_(self.linear_r.bias)
        init.zeros_(self.linear_q.bias)

        init.zeros_(self.linear_b.bias)

        init.zeros_(self.linear_b3.bias)
        init.zeros_(self.linear_b4.bias)

        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    @property
    def device(self):
        return self.linear_r.weight.device

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        batch_size = optimizees.X.numel()
        self.state = (
            # hidden_state
            torch.randn(
                self.layers, batch_size, self.hidden_size
            ).to(self.device),
            # cell_state
            torch.randn(
                self.layers, batch_size, self.hidden_size
            ).to(self.device),
        )
        self.step_size = (step_size if step_size
                          else 0.9999 / optimizees.grad_lipschitz())

        self.step_size = torch.clamp(self.step_size, min=1e-20)

        self.step_size_non_smooth = (step_size if step_size
                                     else 0.9999 / optimizees.sub_grad_lipschitz_non_smooth())

        if kwargs.get('B_step_size', None) is None:
            self.B_step_size = 1.0
        elif kwargs.get('B_step_size', None) == 'B':
            self.B_step_size = 1.0
        elif kwargs.get('B_step_size', None) == 'BsqrtL':
            self.B_step_size = torch.sqrt(self.step_size)
        elif kwargs.get('B_step_size', None) == 'BL':
            self.B_step_size = self.step_size
        elif kwargs.get('B_step_size', None) == 'BLL':
            self.B_step_size = self.step_size * self.step_size

        # self.B_step_size = torch.clamp(self.B_step_size, min=0.0, max=1.0)

        if kwargs.get('C_step_size', None) is None:
            self.C_step_size = 1.0
        elif kwargs.get('C_step_size', None) == 'C':
            self.C_step_size = 1.0
        elif kwargs.get('C_step_size', None) == 'CsqrtL':
            self.C_step_size = torch.sqrt(self.step_size)
        elif kwargs.get('C_step_size', None) == 'CL':
            self.C_step_size = self.step_size
        elif kwargs.get('C_step_size', None) == 'CLL':
            self.C_step_size = self.step_size * self.step_size

        # self.C_step_size = torch.clamp(self.C_step_size, min=0.0, max=1.0)

        self.init_grad_norm_smooth = self._get_init_grad_norm(optimizees)
        self.init_subgrad_norm_c, self.init_subgrad_norm_d = self._get_init_subgrad_norms(
            optimizees)

    def _get_init_grad_norm(self, optimizees: BaseOptimizee):
        smooth_grad = optimizees.get_grad(
            grad_method='smooth_grad',
            compute_grad=self.training,
            retain_graph=self.training,
        )

        init_grad_norm_smooth = torch.clamp(torch.linalg.norm(
            smooth_grad, ord=2, dim=(-2, -1), keepdim=True), min=1e-20)
        return init_grad_norm_smooth

    def _get_init_subgrad_norms(self, optimizees: BaseOptimizee):
        subgrad_nonsmooth = optimizees.get_grad(
            grad_method='subgrad_nonsmooth',
            compute_grad=self.training,
            retain_graph=self.training,
        )

        init_subgrad_norm_c = torch.clamp(torch.linalg.norm(
            subgrad_nonsmooth[0], ord=2, dim=(-2, -1), keepdim=True), min=1e-20)
        init_subgrad_norm_d = torch.clamp(torch.linalg.norm(
            subgrad_nonsmooth[1], ord=2, dim=(-2, -1), keepdim=True), min=1e-20)
        return init_subgrad_norm_c, init_subgrad_norm_d

    def _get_init_inte_norms(self, optimizees: BaseOptimizee):
        smooth_grad = optimizees.get_grad(
            grad_method='smooth_grad',
            compute_grad=self.training,
            retain_graph=self.training,
        )

        subgrad_nonsmooth = optimizees.get_grad(
            grad_method='subgrad_nonsmooth',
            compute_grad=self.training,
            retain_graph=self.training,
        )

        init_subgrad_norm_c = torch.clamp(torch.linalg.norm(
            smooth_grad + subgrad_nonsmooth[0], ord=2, dim=(-2, -1), keepdim=True), min=1e-20)
        init_subgrad_norm_d = torch.clamp(torch.linalg.norm(
            smooth_grad + subgrad_nonsmooth[1], ord=2, dim=(-2, -1), keepdim=True), min=1e-20)
        return init_subgrad_norm_c, init_subgrad_norm_d

    def detach_state(self):
        if self.state is not None:
            self.state = (self.state[0].detach(), self.state[1].detach())

        # self.smooth_grad = None
        # self.nonsmooth_subgrad = None

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model.
        """
        return 'GOMathL2OLH'

    def argmin(self, optimizees, prox_in_P, prox_in):
        return optimizees.prox(
            {'P': prox_in_P, 'X': prox_in}, compute_grad=self.training)

    def forward(
        self,
        optimizees: BaseOptimizee,
        grad_method: str,
        reset_state: bool = False,
        detach_grad: bool = False,
    ):
        """docstrings
        TBA
        """
        # batch_size = optimizees.batch_size

        if self.state is None or reset_state:
            self.reset_state(optimizees)

        smooth_grad = optimizees.get_grad(
            grad_method=grad_method,
            compute_grad=self.training,
            retain_graph=self.training,
        )

        nonsmooth_subgrad = optimizees.get_grad(
            grad_method='subgrad_nonsmooth',
            compute_grad=self.training,
            retain_graph=self.training,
        )

        lstm_input = smooth_grad / self.init_grad_norm_smooth
        lstm_input_c = nonsmooth_subgrad[0] / self.init_subgrad_norm_c
        lstm_input_d = nonsmooth_subgrad[1] / self.init_subgrad_norm_d

        if detach_grad:
            lstm_input = lstm_input.detach()
            lstm_input_c = lstm_input_c.detach()
            lstm_input_d = lstm_input_d.detach()

        # Core update by LSTM.
        lstm_input = lstm_input.flatten().unsqueeze(0).unsqueeze(-1)
        lstm_input_c = lstm_input_c.flatten().unsqueeze(0).unsqueeze(-1)
        lstm_input_d = lstm_input_d.flatten().unsqueeze(0).unsqueeze(-1)
        lstm_in = torch.cat((lstm_input, lstm_input_c, lstm_input_d), dim=2)
        output, self.state = self.lstm(lstm_in, self.state)

        output = Func.relu(self.linear(output))

        R = self.linear_r(output).reshape_as(optimizees.X)
        Q = self.linear_q(output).reshape_as(optimizees.X)

        B = self.linear_b(output).reshape_as(optimizees.X)

        R = self.r_norm(R) * self.r_scale if self.r_use else 1.0
        Q = self.q_norm(Q) * self.q_scale if self.q_use else 1.0

        R = R * self.step_size
        Q = Q * self.B_step_size

        # Set B to be [0, 1]
        B = self.b_norm(B) * self.b_scale if self.b_use else 1.0

        b3 = self.linear_b3(output).reshape_as(optimizees.X)
        b3 = self.b3_norm(b3) * self.b3_scale if self.b3_use else 0.0
        b4 = self.linear_b4(output).reshape_as(optimizees.X)
        b4 = self.b4_norm(b4) * self.b4_scale if self.b4_use else 0.0

        # Apply the update to the iterate
        QVG_HVGNG = Q*optimizees.get_var('VG')
        prox_in = optimizees.X - R*smooth_grad - b3 - QVG_HVGNG

        # Apply the update to the iterate
        prox_out = self.argmin(optimizees, R, prox_in)

        # Recurrent gradient map
        GM = GM = (1/(R+1e-20)) * (optimizees.X - prox_out - b3)

        VG = B*optimizees.get_var('VG') + (1.0-B)*GM - b4

        optimizees.set_var('VG', VG)

        optimizees.X = prox_out

        self.R, self.Q, self.B = R, Q, B
        self.b1, self.b2 = b3, b4
        self.lstm_input = lstm_input
        self.lstm_input_c = lstm_input_c
        self.lstm_input_d = lstm_input_d

        return optimizees


# def test():
#     return True


# if __name__ == "__main__":
#     test()
