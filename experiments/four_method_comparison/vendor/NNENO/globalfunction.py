# -*- coding: utf-8 -*-
"""
Created on Mon April 13 2020

@author: Yue Li
"""

import torch
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import numpy as np
import math
import torch.distributions as tdist
import random
import matplotlib as mpl



#########################################################
'''
function for getting schemes parameters
'''


class Scheme6(object):
    def __init__(self):
        self.num_stancil = 4

    def inputValueToTENO6StencilsL(self, ui):
        """
        function to get values of each stencil
        ------------
        :param ui: input data
        :return: shifted input data
        """
        nx = ui.size(0)
        u_proc = torch.zeros(nx, 6).double()
        # print("u_proc", u_proc.shape)
        # print("ui", ui.shape)

        # print("u_proc", u_proc.shape)
        # print("ui", ui.shape)

        # shift to get the input of each cell
        u_proc[:, 0] = ui.roll(2, 0)
        u_proc[:, 1] = ui.roll(1, 0)
        u_proc[:, 2] = ui.roll(0, 0)
        u_proc[:, 3] = ui.roll(-1, 0)
        u_proc[:, 4] = ui.roll(-2, 0)
        u_proc[:, 5] = ui.roll(-3, 0)
        return u_proc

    def inputValueToTENO6StencilsR(self, ui):
        """
        function to get values of each stencil
        ------------
        :param ui: input data
        :return: shifted input data
        """
        nx = ui.size(0)
        u_proc = torch.zeros(nx, 6).double()

        # shift to get the input of each cell
        u_proc[:, 0] = ui.roll(-3, 0)  # 3
        u_proc[:, 1] = ui.roll(-2, 0)  # 2
        u_proc[:, 2] = ui.roll(-1, 0)  # 1
        u_proc[:, 3] = ui.roll(0, 0)  # 0
        u_proc[:, 4] = ui.roll(+1, 0)  # -1
        u_proc[:, 5] = ui.roll(+2, 0)  # -2

        return u_proc

    def TENO6NN(self, ur, pred1):
        """
        function to calculate the smoothness indicator of the TENO6NN scheme
        ------------
        :param ur: input data
        :param pred: input prediction of the smoothness of the stencil
        :param st: the index of input stencil
        :return: the label of the corresponding stencil
        """
        # values
        v1 = ur[:, 0]  # -2
        v2 = ur[:, 1]  # -1
        v3 = ur[:, 2]  # 0
        v4 = ur[:, 3]  # 1
        v5 = ur[:, 4]  # 2
        v6 = ur[:, 5]  # 3

        target = torch.ones(v1.size()).double() * np.float64(0.5)

        # nn prediction 4stencil
        bb4 = torch.lt(pred1[:,3], target).int()
        bb3 = torch.lt(pred1[:,2], target).int()
        bb2 = torch.lt(pred1[:,1], target).int()
        bb1 = torch.lt(pred1[:,0], target).int()

        variation1 = -1.0 / 6.0 * v2 + 5.0 / 6.0 * v3 + 2.0 / 6.0 * v4 - v3
        variation2 = 2. / 6. * v3 + 5. / 6. * v4 - 1. / 6. * v5 - v3
        variation3 = 2. / 6. * v1 - 7. / 6. * v2 + 11. / 6. * v3 - v3
        variation4 = 3. / 12. * v3 + 13. / 12. * v4 - 5. / 12. * v5 + 1. / 12. * v6 - v3


        a1 = 0.4294317718960898 * bb1
        a2 = 0.1727270875843552 * bb2
        a3 = 0.0855682281039113 * bb3
        a4 = 0.312272912415645 * bb4

        w1 = a1 / (a1 + a2 + a3 + a4)
        w2 = a2 / (a1 + a2 + a3 + a4)
        w3 = a3 / (a1 + a2 + a3 + a4)
        w4 = a4 / (a1 + a2 + a3 + a4)

        flux = v3 + w1 * variation1 + w2 * variation2 + w3 * variation3 + w4 * variation4

        return flux


#########################################################
'''
function for make a multi wave ICs
'''


def funcG(x, theta, zeta):
    beta = -math.log10(2.) / 36.0 / theta / theta

    return torch.exp(beta * (x - zeta) ** 2)


def funcF(x, alpha, a):
    y = 1.0 - alpha * alpha * (x - a) ** 2
    y = torch.clamp(y, min=0.0)
    y = torch.sqrt(y)
    return y


def makeMultiWave():
    a = 0.5
    theta = 0.005
    zeta = -0.7
    alpha = 10
    L = 2.0
    def IC(x):
        f = 0 * x
        one = torch.ones(f.size(), dtype=torch.double)
        oneL = torch.ones(f.size(), dtype=torch.double) * L

        f = torch.where(np.logical_and(x+1.0 < (0.2 * L), x+1.0 >= (0.1 * L)), \
                        1 / 6 * (funcG(x+1.0 - oneL * 0.5, theta, zeta - theta) + \
                                 funcG(x+1.0 - oneL * 0.5, theta, zeta + theta) + \
                                 4 * funcG(x+1.0 - oneL * 0.5, theta, zeta)),
                        f)
        f = torch.where(np.logical_and(x+1.0 < (0.4 * L), x+1.0 >= (0.3 * L)), one, f)

        f = torch.where(np.logical_and(x+1.0 < (0.6 * L), x+1.0 >= (0.5 * L)), \
                        one - torch.abs(10 * (x+1.0 - oneL * 0.55)), f)

        f = torch.where(np.logical_and(x+1.0 < (0.8 * L), x+1.0 >= (0.7 * L)), \
                        1 / 6 * (funcF(x+1.0 - oneL * 0.5, alpha, a - theta) + \
                                 funcF(x+1.0 - oneL * 0.5, alpha, a + theta) + \
                                 4 * funcF(x+1.0 - oneL * 0.5, alpha, a)),
                        f)

        return f

    return IC

def makeICdsc():
    def IC(x):
        # np.random.seed(N)
        f = 0 * x
        L=2.0
        for j in range(0, 5):
            f = f + torch.rand(1, dtype=torch.double) * torch.sin(
                2 * j * np.pi * (x - torch.rand(1, dtype=torch.double)) / L)
        dscc = 1+4*torch.rand(1,dtype=torch.double)
        # f = f + (x > (L / 2)).double() * (5 - 10 * torch.rand(1, dtype=torch.double))
        f = f + (x>(L/2)).double()*dscc*np.random.choice((-1, 1))
        return f

    return IC
def makeICMdsc():
    def IC(x):
        # np.random.seed(N)
        L = 2.0
        f = 0 * x
        for j in range(0, 5):
            f = f + torch.rand(1, dtype=torch.double) * torch.sin(
                2 * j * np.pi * (x - torch.rand(1, dtype=torch.double)) / L)
        dscc = 1+4*torch.rand(1,dtype=torch.double)
        f = f + (x > (L / (1+j*4))).double() * (5 - 10 * torch.rand(1, dtype=torch.double))
        # f = f + (x>(L/2)).double()*dscc*np.random.choice((-1, 1))
        return f

    return IC

def Shuosher():
    def IC(x):
        # np.random.seed(N)
        f = 0 * x
        L=2.0
        f = 2./3.* torch.sin(6 * x* np.pi) + 1./4. * torch.sin(1.6 * x* np.pi)
        return f

    return IC
def Compundwave():
    def IC(x):
        # np.random.seed(N)   range [-4, 4]
        f = 0 * x
        one = torch.ones(f.size(), dtype=torch.double)

        f = torch.where(np.logical_and(abs(x) <= 4.0, abs(x) >= 1.0), \
                        torch.sin(x* np.pi), f)
        f = torch.where(np.logical_or(np.logical_and(x <= (-0.5), x > (-1.0)),
                        np.logical_and(x <= (0.5), x > (0.0))), 3.0*one, f)

        f = torch.where(np.logical_and(x <= 0.0, x > -0.5), 1.0*one, f)

        f = torch.where(np.logical_and(x <= 1.0, x > 0.5), 2.0*one, f)

        return f

    return IC
def shockcollision():
    def IC(x):
        # np.random.seed(N)   range{0, 1} t_terminate = 0.1
        f = 0 * x
        one = torch.ones(f.size(), dtype=torch.double)

        f = torch.where(np.logical_and(x <= 0.2, x >= 0.0), 10.0*one, f)

        f = torch.where(np.logical_and(x <= 0.4, x > 0.2), 6.0*one, f)

        f = torch.where(np.logical_and(x <= 0.6, x > 0.4), 0.0*one, f)

        f = torch.where((x > 0.6), -4.0*one, f)

        return f

    return IC
def sine():
    def IC(x):
        # np.random.seed(N)   range{0, 2} t_terminate = 1.5/np.pi
        # resolution
        f = 0 * x
        one = torch.ones(f.size(), dtype=torch.double)

        f = torch.sin(1.0*x* np.pi)
        # f = torch.where(np.logical_and(x <=5.0/6.0, x >= 1.0/6.0), torch.sin(6.0*x* np.pi), f)

        return f

    return IC

