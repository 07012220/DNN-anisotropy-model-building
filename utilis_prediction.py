""" =====================================================================================
                         Copyright (c) 2020 You Nan, n.you@u.nus.edu                      
    ===================================================================================== """
from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def Hudson_Cheng_model(c33, c44, pred, poro):
    """

    :param c33: measured C33
    :param c44: measured C44
    :param pred: DNN predictions of the matrix moduli and aspect ratio of thin cracks, [k0, mu0, alpha]
    :param poro: measured porosity
    :return: anisotropy parameters, DNN predicted crack density and C33, C44
    """

    k_fluid = 2.2
    k0 = pred[:, 0]
    mu0 = pred[:, 1]
    alpha = pred[:, 2]
    HS_k = k0 + poro / (1.0 / (k_fluid - k0) + (1.0 - poro) / (k0 + 4.0 / 3.0 * mu0))
    HS_mu = mu0 + poro / (1.0 / (0 - mu0) + 2.0 * (1.0 - poro) * (k0 + 2.0 * mu0) / (5.0 * mu0 * (k0 + 4.0 / 3.0 * mu0)))
    c33_0 = HS_k + 4.0 / 3.0 * HS_mu
    c44_0 = HS_mu
    mu = c44_0
    lamda = c33_0 - 2.0 * c44_0
    K = k_fluid * (lamda + 2.0 * mu) / (np.pi * alpha * mu * (lamda + mu))
    U3 = 4.0 / 3.0 * (lamda + 2.0 * mu) / (lamda + mu) / (1.0 + K)
    U1 = 16.0 / 3.0 * (lamda + 2.0 * mu) / (3.0 * lamda + 4.0 * mu)
    c33_1 = c33 - c33_0
    c44_1 = c44 - c44_0
    left_c33 = - c33_1 * mu / ((lamda + 2.0 * mu) ** 2.0)
    left_c44 = - c44_1 / mu
    crackd_c33 = left_c33 / U3
    crackd_c44 = left_c44 / U1
    crackd = 0.5 * (crackd_c33 + crackd_c44)
    crackd[crackd < 0] = 0

    # compute the full elastic tensor
    c11_0 = c33_0
    c13_0 = lamda
    c11_1 = - lamda ** 2.0 / mu * crackd * U3
    c13_1 = - lamda * (lamda + 2.0 * mu) * crackd * U3 / mu
    c33_1_dnn = - (lamda + 2.0 * mu) ** 2.0 * crackd * U3 / mu
    c44_1_dnn = -mu * crackd * U1
    c11 = c11_0 + c11_1
    c13 = c13_0 + c13_1
    c66 = c44_0
    c33_dnn = c33_0 + c33_1_dnn
    c44_dnn = c44_0 + c44_1_dnn

    epsilon = (c11 - c33_dnn) / (2.0 * c33_dnn)
    gamma = (c66 - c44_dnn) / (2.0 * c44_dnn)
    delta = 1.0 / (2.0 * c33_dnn * (c33_dnn - c44_dnn)) * ((c13 + c44_dnn) ** 2.0 - (c33_dnn - c44_dnn) ** 2.0)
    return epsilon, gamma, delta, crackd, c33_dnn, c44_dnn


def plot_results(c33, c44, c33dnn, c44dnn, name):
    """

    :param name: data name, str
    :param c33: measured c33
    :param c44: measured c44
    :param c33dnn: DNN predicted c33
    :param c44dnn: DNN predicted c44
    :return: 0
    """

    plt.figure()
    upper_bound = np.ceil(max(max(c33), max(c33dnn))/10) * 10
    plt.plot([0, upper_bound], [0, upper_bound], 'k')
    plt.plot(c33, c33dnn, 'r.', label='C33')
    plt.plot(c44, c44dnn, 'b.', label='C44')
    plt.box(True)
    plt.xlabel('Measured Cij (GPa)')
    plt.ylabel('DNN predicted Cij (GPa)')
    plt.legend()
    plt.xlim([0, upper_bound])
    plt.ylim([0, upper_bound])
    plt.title('Comparison of DNN predicted and measured C33 and C44 (' + name + ')')
    return 0
