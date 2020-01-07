from utilis_prediction import *
import os
from scipy.io import loadmat, savemat

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load the features of the field data and the lab data
x_CN = np.loadtxt('data/features_ChineseWell.csv', delimiter=',', skiprows=1)
x_US = np.loadtxt('data/features_USWell.csv', delimiter=',', skiprows=1)
x_Lab = np.loadtxt('data/features_LabData.csv', delimiter=',', skiprows=1)

# load the distribution parameters of the training data
dist_training_data = loadmat('data/dist_trainingData.mat')
# Normalize the field data using the distribution of the training data since they lie well
# within the training data distribution
x_CN_norm = (x_CN - dist_training_data['x_avr']) / dist_training_data['x_std']
x_US_norm = (x_US - dist_training_data['x_avr']) / dist_training_data['x_std']

# load the distribution parameters of the lab data
dist_lab_data = loadmat('data/dist_labData.mat')
# Normalize the lab data suing their own distribution since they present significantly different
# distribution in both ranges and patterns from that of the training data
std_lab = dist_lab_data['x_std']
std_lab[std_lab == 0] = 1
x_Lab_norm = (x_Lab - dist_lab_data['x_avr']) / std_lab

print('The feature shape of the selected sublayer of the Chinese well logs: {}'.format(x_CN_norm.shape))
print('The feature shape of the collected lab data: {}'.format(x_Lab_norm.shape))
print('The feature shape of the selected sublayer of the US well logs: {}'.format(x_US_norm.shape))

# load model
model = load_model('ckpt/best_weight[64,128,256,64].hdf5')

# predict
y_CN_norm = model.predict(x_CN_norm, batch_size=None, verbose=1, steps=None)
y_Lab_norm = model.predict(x_Lab_norm, batch_size=None, verbose=1, steps=None)
y_US_norm = model.predict(x_US_norm, batch_size=None, verbose=1, steps=None)

# scale the raw predictions to normal ranges
# benchmark the field data with the distribution of the training data
# benchmark the lab data with their own distribution
y_CN = y_CN_norm * dist_training_data['y_std'] + dist_training_data['y_avr']
y_US = y_US_norm * dist_training_data['y_std'] + dist_training_data['y_avr']
y_Lab = y_Lab_norm * dist_lab_data['y_std'] + dist_lab_data['y_avr']

# calculate anisotropy parameters based on chosen Hudson-Cheng model
epsilonCN, gammaCN, deltaCN, crackdCN, c33_dnnCN, c44_dnnCN = Hudson_Cheng_model(
    x_CN[:, 4], x_CN[:, 5], y_CN, x_CN[:, 0])
epsilonUS, gammaUS, deltaUS, crackdUS, c33_dnnUS, c44_dnnUS = Hudson_Cheng_model(
    x_US[:, 4], x_US[:, 5], y_US, x_US[:, 0])
epsilonLab, gammaLab, deltaLab, crackdLab, c33_dnnLab, c44_dnnLab = Hudson_Cheng_model(
    x_Lab[:, 4], x_Lab[:, 5], y_Lab, x_Lab[:, 0])

plot_results(x_CN[:, 4], x_CN[:, 5], c33_dnnCN, c44_dnnCN, 'CN')

plot_results(x_US[:, 4], x_US[:, 5], c33_dnnUS, c44_dnnUS, 'US')

plot_results(x_Lab[:, 4], x_Lab[:, 5], c33_dnnLab, c44_dnnLab, 'Lab')

plt.show()

# save predictions
savemat('pred/predictions_CN.mat',
        {'k0': y_CN[:, 0], 'mu0': y_CN[:, 1], 'alpha': y_CN[0, 2], 'epsilon': epsilonCN, 'gamma': gammaCN,
         'delta': deltaCN})
savemat('pred/predictions_US.mat',
        {'k0': y_US[:, 0], 'mu0': y_US[:, 1], 'alpha': y_US[0, 2], 'epsilon': epsilonUS, 'gamma': gammaUS,
         'delta': deltaUS})
savemat('pred/predictions_Lab.mat',
        {'k0': y_Lab[:, 0], 'mu0': y_Lab[:, 1], 'alpha': y_Lab[0, 2], 'epsilon': epsilonLab, 'gamma': gammaLab,
         'delta': deltaLab, 'c33': c33_dnnLab, 'c44': c44_dnnLab})
