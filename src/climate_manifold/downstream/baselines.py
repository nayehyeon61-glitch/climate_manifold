"""Recurrent MLP and RK4 Neural ODE baselines with matching field contracts."""
import math
import torch
from torch import nn
from ..nn import mlp


def calendar_features(origin_ns, like):
    # Deterministic calendar conditioning; no observations after origin are used.
    hours = origin_ns.to(torch.float64) / 3.6e12
    angles = torch.stack((2*math.pi*(hours % 24)/24, 2*math.pi*(hours % (365.25*24))/(365.25*24)), -1)
    return torch.cat((angles.sin(), angles.cos()), -1).to(like)


class HistoryPredictor(nn.Module):
    def __init__(self, dimension, history_steps, hidden=128, information_dim=0, kind='mlp', substeps=2):
        super().__init__()
        if kind not in ('mlp', 'neural_ode') or substeps < 1:
            raise ValueError('Invalid baseline kind/ODE substeps')
        self.kind, self.substeps, self.information_dim = kind, substeps, information_dim
        self.context = mlp(history_steps*dimension + information_dim + 4, hidden, hidden)
        self.field = mlp(dimension + hidden + 1, hidden, dimension)

    def forward(self, history, lead_hours, origin_ns, information=None):
        items = [history.flatten(1), calendar_features(origin_ns, history)]
        if self.information_dim:
            if information is None or information.shape != (len(history), self.information_dim):
                raise ValueError('Origin information is required for this predictor')
            items.append(information)
        context = self.context(torch.cat(items, -1))
        def rhs(state, time_days):
            time = state.new_full((len(state), 1), float(time_days))
            return self.field(torch.cat((state, context, time), -1))
        state, previous, result = history[:, -1], 0., []
        for hour in lead_hours.tolist():
            final = hour/24
            if self.kind == 'mlp':
                state = state + (final-previous)*rhs(state, previous)
            else:
                dt = (final-previous)/self.substeps
                for i in range(self.substeps):
                    t = previous+i*dt
                    k1 = rhs(state, t)
                    k2 = rhs(state+dt*k1/2, t+dt/2)
                    k3 = rhs(state+dt*k2/2, t+dt/2)
                    k4 = rhs(state+dt*k3, t+dt)
                    state = state + dt*(k1+2*k2+2*k3+k4)/6
            result.append(state); previous = final
        return torch.stack(result, 1), None
