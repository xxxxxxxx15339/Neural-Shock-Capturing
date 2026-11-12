"""
Created on Mon April 13 2020

@author: Yue Li
"""

from decimal import Decimal
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from globalfunction import Scheme6, makeMultiWave,makeICdsc,makeICMdsc,Shuosher,sine


plt.close('all')  # close all open figures

######################## Define NN model ##########################
'''
class for NN model definition
'''

class Model(nn.Module):
    def __init__(self, input_size, output_size):

        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64).double()
        self.fc2 = torch.nn.Linear(64, 32).double()
        self.fc3 = torch.nn.Linear(32, 16).double()
        self.fc4 = torch.nn.Linear(16, 8).double()
        self.fc5 = torch.nn.Linear(8, output_size).double()
        # self.fc4 = torch.nn.Linear(32, 16).double()
        # self.fc5 = torch.nn.Linear(16, output_size).double()
        # self.tanh    = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        # self.leakrelu = torch.nn.LeakyReLU()
        # self.m = nn.Softmax(dim=0)
        self.m = nn.Sigmoid()

    def forward(self, ui):

        hidden1 = self.relu(self.fc1(ui))
        hidden2 = self.relu(self.fc2(hidden1))
        hidden3 = self.relu(self.fc3(hidden2))
        hidden4 = self.relu(self.fc4(hidden3))
        hidden5 = self.fc5(hidden4)
        wout= self.m(hidden5)
        return wout

input_size = 6
output_size = 4
Ll = -1.0
Lr = 1.0
L = Lr - Ll
num_ghost = 4
CFL = np.float64(0.4)
t_terminate = np.float64(6.0)
resolution = 150
dx = np.float64(L / resolution)
dt = np.float64(CFL * dx)
dt_plot = 0.5
nt = int(t_terminate / dt)
nt_out = int(dt_plot / dt)

L += num_ghost * 2 * dx
d_start = Ll - num_ghost * dx
d_end = Lr + num_ghost * dx
xx = torch.linspace(d_start, d_end, int(resolution+2*num_ghost) + 1, dtype=torch.double)
xx = xx[:-1]
xx = xx + dx / 2.0

# initialize 1D mesh
yy = torch.zeros(xx.shape, dtype=torch.double)
ll = torch.zeros(xx.shape, dtype=torch.double)

# initialize numerical scheme
scheme = Scheme6()

# initialize ic generator
data_ics = makeMultiWave()
# ui, positive_only = data_ics.get_type(10)(xx,scheme,0) #1 9 11
ui = data_ics(xx)
ui_solt = ui
ui_teno = ui

# fig, axs = plt.subplots(2, 2, figsize=(15, 10))
# plt.show(False)
# plt.draw()

# initialize nn model
model1 = Model(input_size=input_size, output_size=output_size)

# load model
model1.load_state_dict(torch.load('mytrainmodel1'))

# time marching
run_t = np.float64(0.0)

ui_1 = torch.zeros(ui.shape, dtype=torch.double)
ui_1_teno = torch.zeros(ui.shape, dtype=torch.double)

print(t_terminate)
print(run_t)
print(dt)

for k in range(0, nt):
    if run_t < t_terminate:
        if (t_terminate - run_t) < dt:
            dt = t_terminate - run_t
        for rk in range(0, 3):
            # impose Boundary condition
            ui[0] = ui[resolution + 0]
            ui[1] = ui[resolution + 1]
            ui[2] = ui[resolution + 2]
            ui[3] = ui[resolution + 3]
            ui[resolution + 4] = ui[4]
            ui[resolution + 5] = ui[5]
            ui[resolution + 6] = ui[6]
            ui[resolution + 7] = ui[7]

            ui_teno[0] = ui_teno[resolution + 0]
            ui_teno[1] = ui_teno[resolution + 1]
            ui_teno[2] = ui_teno[resolution + 2]
            ui_teno[3] = ui_teno[resolution + 3]

            ui_teno[resolution + 4] = ui_teno[4]
            ui_teno[resolution + 5] = ui_teno[5]
            ui_teno[resolution + 6] = ui_teno[6]
            ui_teno[resolution + 7] = ui_teno[7]

            # calculate the flux
            flux_L_in_ori = scheme.inputValueToTENO6StencilsL(ui)
            flux_L_in = scheme.inputValueToTENO6StencilsL(ui)

            # normalize data
            yy1 = flux_L_in[:, 0]
            yy2 = flux_L_in[:, 1]
            yy3 = flux_L_in[:, 2]
            yy4 = flux_L_in[:, 3]
            yy5 = flux_L_in[:, 4]
            yy6 = flux_L_in[:, 5]

            yymax = torch.max(torch.abs(yy1), torch.abs(yy2))
            yymax = torch.max(yymax, torch.abs(yy3))
            yymax = torch.max(yymax, torch.abs(yy4))
            yymax = torch.max(yymax, torch.abs(yy5))
            yymax = torch.max(yymax, torch.abs(yy6))
            yymax = torch.max(yymax, torch.ones(yy3.size()).double())

            flux_L_in[:, 0] /= yymax
            flux_L_in[:, 1] /= yymax
            flux_L_in[:, 2] /= yymax
            flux_L_in[:, 3] /= yymax
            flux_L_in[:, 4] /= yymax
            flux_L_in[:, 5] /= yymax

            # apply trained nn model
            pred1 = model1(flux_L_in).view(-1,4)

            flux = scheme.TENO6NN(flux_L_in_ori, pred1)

            # flux_teno = scheme.WENO6cu6(flux_L_in_teno)

            LU = -dt * (flux - flux.roll(1, 0)) / dx

            # #### update the solution
            if rk == 0:
                ui_1 = ui
                ui = ui + LU

            elif rk == 1:
                # print("ui_1", ui_1)
                # print("ui", ui)
                ui = 0.25 * ui + 0.75 * ui_1 + 0.25 * LU
                # print("rk1", ui)
            else:
                ui = 2. / 3. * ui + 1. / 3. * ui_1 + 2. / 3. * LU
                # print("rk2", ui.shape)
                # print("rk2", ui)

        if k%nt_out == 0:
            xxx = xx - run_t%(Lr-Ll)
            ii = (xx-run_t%(Lr-Ll)) <= -1.0
            xxx[ii] = (Lr-Ll) + xx[ii] - run_t%(Lr-Ll)
            solt = data_ics(xxx)
            # solt = data_ics.get_type(10)(xxx,scheme)

            # mpl.use('MacOSX')
            pred_1=pred1[:,0].detach()
            pred_2=pred1[:,1].detach()
            pred_3=pred1[:,2].detach()
            pred_4=pred1[:,3].detach()

            target = torch.ones(np.shape(pred_1)) * np.float64(0.5)

            pred_11 = torch.gt(pred_1, target).int()
            pred_22 = torch.gt(pred_2, target).int()
            pred_33 = torch.gt(pred_3, target).int()
            pred_44 = torch.gt(pred_4, target).int()


            plt.clf()
            plt.plot(xx[num_ghost:-num_ghost-1],solt[num_ghost:-num_ghost-1], '--b')
            plt.plot(xx[num_ghost:-num_ghost],ui[num_ghost:-num_ghost], '-r')
            plt.xlabel('$x$')
            plt.ylabel('u$')
            plt.legend(('exact','teno6NN'))
            plt.pause(0.05)

        run_t += dt

        # print(ui)
        print("run_t", run_t)
        #########################################################
    else:
        break



