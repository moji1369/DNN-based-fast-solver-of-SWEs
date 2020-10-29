import os
import warnings
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from keras import optimizers
from scipy.io import savemat, loadmat
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout, Dense, Activation
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras import metrics, losses
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from scipy.linalg import svd
from keras.models import load_model

warnings.filterwarnings("ignore")


def generate_model(input_dim, output_dim, n_layer, n_hidden,
                   optimizer_meth, act_fcn, learning_rate=0.001, reg=0.01):
    """
    input_dim:           the size of input vector
    output_dim:          the size of output vector ('k' in the problem set)
    n_layer:             number of hidden layers in the network
    n_hidden:            a list of number of hidden units in each hidden layer. 
                         for example, for n_hidden=5, this can be [200, 100, 45]
    optimizer_meth:      the optimization method used for training
    act_fcn              the type of activation function in the hidden layers
    learning_rate        Learning rate used in the training phase
    reg                  L2 regularization coefficient
    """
    model = Sequential()  
    model.add(Dense(units=n_hidden[0], input_dim=input_dim,
                    kernel_regularizer=regularizers.l2(reg)))
    model.add(BatchNormalization())
    model.add(Activation(act_fcn))
    #### ToDo build your model here and compile it.
    """ hint: Keep the parameters for the hidden layers the same, and use for loop to add the layers to the 
              network. Add the last layer (output layer) outside the for loop since it has a different
              activation function. 
        
    """
    for i in range(1, n_layer-1):
        model.add(Dense(units=n_hidden[i], input_dim=n_hidden[i-1], kernel_regularizer=regularizers.l2(reg)))
        model.add(BatchNormalization())
        model.add(Activation(act_fcn))
    
    model.add(Dense(units=n_hidden[-1], activation='linear', kernel_regularizer=regularizers.l2(reg)))
    
    model.compile(loss='mean_squared_error', optimizer=optimizer_meth, metrics=['mse'])  
    
    return model


def train(model, X_train, y_train, batch_s, n_epochs, verbose=1, validation_split=.2):
    """
    model:       the DNN model (the output of Generate_model)
    X_train:     input of the network (flow velocity measurements)
    y_train:     output of the network (k largest eigenvalues)
    batch_size:  batch size parameter
    epochs:      number of epochs used to train the network
    validation_split:  validation/train split ration
    """
    
    #### ToDo:  use the training data and train your model. 
    #### hint: Plot the loss function for the training and validation set to observe your network performance.
    history = model.fit(X_train, y_train, batch_size=batch_s, epochs=n_epochs, verbose=1, validation_split=.2)
    plt.plot(range(n_epochs), history.history['val_loss'], range(n_epochs), history.history['loss'])
    
    return history.history['val_loss']


def post_process(data, U):
    nx = [501, 41]
    n_edge = 12
    a = np.zeros((nx[1], nx[0]))
    for i in range(data.shape[0]):
        a[:, i:i+n_edge] += np.dot(U, data[i, :]).reshape(nx[1], -1)
    for i in range(n_edge):
        a[:, i] = a[:, i]/(i+1)
        a[:, nx[0]-i-1] = a[:, nx[0]-i-1]/(i+1)

    a[:, n_edge: nx[0]-n_edge] = a[:, n_edge: nx[0]-n_edge]/n_edge
    return a


def plt_im_tri(depth, fig_name, fig_title= 'title', show_file=True, vmin_=21.0, vmax_=29.0):
    mesh = loadmat("mesh.mat")
    triangles = mesh['triangles']
    meshnode = mesh['meshnode']
    matplotlib.rcParams.update({'font.size': 16})

    offsetx = 220793.
    offsety = 364110.
    fig_index = 1
    plt.figure(fig_index, figsize=(10., 10.), dpi=100)
    fig_index += 1
    ax = plt.gca()
    im = plt.tripcolor(meshnode[:, 0]*0.3010-offsetx, meshnode[:, 1]*0.3010-offsety,
                       triangles, depth*0.301, cmap=plt.get_cmap('jet'), vmin=vmin_, vmax=vmax_, label='_nolegend_')
    ax.set_xlabel("Easting [m]")
    ax.set_ylabel("Northing [m]")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([0., 1000., 0., 530.])
    plt.xticks(np.arange(0., 1000.+10., 200.0))
    plt.yticks(np.arange(0., 530.+10., 200.0))
    plt.title(fig_title)
    cbar = plt.colorbar(im, fraction=0.025, pad=0.05)
    #cbar.set_label('Velocity [m/s]')
    cbar.set_label('Elevation [m]')
    plt.rcParams['axes.axisbelow'] = True
    plt.rc('axes', axisbelow=True)
    plt.grid()
    ax.set_axisbelow(True)
    plt.tight_layout()
    ax.axis('off')
    ax.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    plt.savefig(fig_name, bbox_inches = 'tight',pad_inches = 0)
    if show_file:
        plt.show()

      
def plot_result(model, mean_X=None, std_X=None, mean_Y=None, std_Y=None):
    data_red_river = loadmat('P2_test.mat')
    X = data_red_river['X']
    Y = data_red_river['Y']
    U = data_red_river['U']
    X_P = (X-mean_X)/std_X
    pred = model.predict(X_P)
    Y_P = pred*std_Y+mean_Y
    river_prof = post_process(Y_P, U)
    # river_prof = post_process(Y, U)
    plt_im_tri(river_prof.ravel(), 'predicted_river.jpg', False)


def plot_result_xx(model, H, mean_X=None, std_X=None, mean_Y=None, std_Y=None):
    data_red_river = loadmat('P2_test.mat')
    x = data_red_river['X']
    X = np.dot(H, x.T).T
    Y = data_red_river['Y']
    U = data_red_river['U']
    X_P = (X-mean_X)/std_X
    pred = model.predict(X_P)
    Y_P = pred*std_Y+mean_Y
    river_prof = post_process(Y_P, U)
    # river_prof = post_process(Y, U)
    plt_im_tri(river_prof.ravel(), 'predicted_river_cgan.png', False)

    
def generate_Q(xmin, dx, nx, L):
    length_x = [nx[0]*dx[0], nx[1]*dx[1]]
    xr = np.linspace(xmin[0]+.5*dx[0],  xmin[0]+length_x[0]-.5*dx[0], nx[0])
    yr = np.linspace(xmin[1]+.5*dx[1],  xmin[1]+length_x[1]-.5*dx[1], nx[1])
    x, y = np.meshgrid(xr, yr)
    x1, x2 = np.meshgrid(x, x)
    y1, y2 = np.meshgrid(y, y)
    distance_x = (x1 - x2)**2
    distance_y = (y1 - y2)**2
    distance = distance_x/(L[0]**2) + distance_y/(L[1]**2)
    Q = np.exp(-distance)
    return Q


def project_to_pc(dep, u):
    return np.dot(u.T, dep)


def create_sub_samples(prof_vel, prof_dep, z_f, Q_b, rec_len, nx):
    dx = 3
    n_samp_line = 5
    n_s_x = rec_len//dx
    n_prof = prof_vel.shape[1] #200
    n_sample_per_prof = (nx[0]-rec_len+1) #501-11+1
    n_sample = n_sample_per_prof*n_prof #491*200
    X = np.zeros((2*n_s_x*n_samp_line+2, n_sample)) #2*(11/3)*5+2 , 491*200
    vel_x, vel_y, _ = xy_vel_sep(prof_vel)
    out_h = np.zeros((nx[1]*rec_len, n_sample)) #41*11 , 491*200
    out_x = np.zeros((nx[1]*rec_len, n_sample))
    out_y = np.zeros((nx[1]*rec_len, n_sample))
    kk = 0
    sample_x = [4, 12, 20, 28, 36] #+8
    sample_y = [1, 4, 7, 10] #+3
    for i in range(n_prof):
        dom_vel_x = np.reshape(vel_x[:, i], (nx[1], nx[0])) #41*501
        dom_vel_y = np.reshape(vel_y[:, i], (nx[1], nx[0]))
        dom = np.reshape(prof_dep[:, i], (nx[1], nx[0]))
        for k in range((rec_len)//2, nx[0]-((rec_len)//2)): #11/2 : 491- 11/2
            meas_x = dom_vel_x[sample_x, k-(rec_len)//2:k+(rec_len)//2]
            meas_y = dom_vel_y[sample_x, k-(rec_len)//2:k+(rec_len)//2]
            for j in range(1):
                m_x = meas_x[:, [j, j+3, j+6, j+9]]
                m_y = meas_y[:, [j, j+3, j+6, j+9]]
                measurement = np.concatenate((m_x.ravel(), m_y.ravel()))
                X[:,kk] = np.concatenate((measurement.ravel(),[z_f[i], Q_b[i]]))
                out_h[:, kk] = dom[:, k-(rec_len)//2:k+(rec_len)//2].ravel()
                out_x[:, kk] = dom_vel_x[:, k-(rec_len)//2:k+(rec_len)//2].ravel()
                out_y[:, kk] = dom_vel_y[:, k-(rec_len)//2:k+(rec_len)//2].ravel()
                kk += 1

    output = {0: out_h, 1:out_x, 2:out_y}
    return X, output




def create_sub_samples_test(prof_vel, prof_dep, rec_len, nx, u):

    dx=3
    n_samp_line = 5
    n_s_x = rec_len//dx
    n_prof = prof_vel.shape[1]
    n_sample_per_prof = (nx[0]-rec_len+1)
    n_sample = n_sample_per_prof*n_prof
    X = np.zeros((n_s_x*n_samp_line+1, n_sample)) #11/2*5+1 , 491*200
    # y = np.zeros((rec_len*nx[1], n_sample))
    y = np.zeros((u.shape[1], n_sample))
    kk = 0
    for i in range(n_prof):
        dom_vel = np.reshape(prof_vel[:, i], (nx[1], nx[0])) #41*501
        dom = np.reshape(prof_dep[:, i], (nx[1], nx[0]))
        for k in range((rec_len)//2, nx[0]-((rec_len)//2)):
            meas = dom_vel[[4,12,20,28,36], k-(rec_len)//2:k+(rec_len)//2]
            j = dx-1-(kk%dx)
            measurement = meas[:, [j, j+3, j+6,j+9]]
            X[:, kk] = np.concatenate((measurement.ravel(),[j]))
            dep=dom[:, k-(rec_len)//2:k+(rec_len)//2].ravel()
            y[:, kk] = project_to_pc(dep, u)
            kk += 1

    return X, y


def xy_vel_sep(prof_vel):

    N, n_prof = prof_vel.shape
    x_vel = np.zeros((N//2, n_prof))
    y_vel = np.zeros((N//2, n_prof))
    mag_vel = np.zeros((N//2, n_prof))
    for i in range(n_prof):
        x_vel[:, i] = prof_vel[0::2, i]
        y_vel[:, i] = prof_vel[1::2, i]
        mag_vel[:, i] = np.sqrt(x_vel[:, i]**2+y_vel[:, i]**2)

    return x_vel, y_vel, mag_vel