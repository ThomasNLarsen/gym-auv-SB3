import gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class RadarCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space) of dimension (N_sensors x 1)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    #def __init__(self, observation_space: gym.spaces.Box, sensor_dim: int = 180, features_dim: int = 32, kernel_overlap: float = 0.05):
    def __init__(self, observation_space: gym.spaces.Box, sensor_dim : int = 180, features_dim: int = 12, kernel_overlap : float = 0.25):
        super(RadarCNN, self).__init__(observation_space, features_dim=features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        # Adjust kernel size for sensor density. (Default 180 sensors with 0.05 overlap --> kernel covers 9 sensors.
        self.in_channels = observation_space.shape[0]  # 180
        self.kernel_size = round(sensor_dim * kernel_overlap)  # 45
        self.kernel_size = self.kernel_size + 1 if self.kernel_size % 2 == 0 else self.kernel_size  # Make it odd sized
        #self.padding = (self.kernel_size - 1) // 2  # 22
        self.padding = self.kernel_size // 3  # 15
        self.stride = self.padding
        #print("RADAR_CNN CONFIG")
        #print("\tIN_CHANNELS =", self.in_channels)
        #print("\tKERNEL_SIZE =", self.kernel_size)
        #print("\tPADDING     =", self.padding)
        #print("\tSTRIDE      =", self.stride)
        self.cnn = nn.Sequential(
            # in_channels: sensor distance, obst_velocity_x, obst_velocity_y
            nn.Conv1d(in_channels=self.in_channels, out_channels=1, kernel_size=self.kernel_size, padding=self.padding,
                      padding_mode='circular', stride=self.stride),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        self.n_flatten = 0
        sample = th.as_tensor(observation_space.sample()).float()
        print("Observation space - sample shape:", sample.shape)
        sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        with th.no_grad():
            print("RadarCNN initializing, CNN input is", sample.shape, "and", end=" ")
            flatten = self.cnn(sample)
            self.n_flatten = flatten.shape[1]
            print("output is", flatten.shape)

        self.linear = nn.Sequential(nn.Linear(self.n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)
        #return self.linear(self.cnn(observations))
        #return self.cnn(observations)

    def get_features(self, observations: th.Tensor) -> list:
        feat = []
        out = observations
        for layer in self.cnn:
            out = layer(out)
            if not isinstance(layer, nn.ReLU):
                feat.append(out.cpu().detach().numpy())

        #for layer in self.linear:
        #    out = layer(out)
        #    if not isinstance(layer, nn.ReLU):
        #        feat.append(out.cpu().detach().numpy())

        return feat

    def get_activations(self, observations: th.Tensor) -> list:
        feat = []
        out = observations
        for layer in self.cnn:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                feat.append(out)

        for layer in self.linear:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                feat.append(out.detach().numpy())

        return feat

class NavigatioNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 6):
        super(NavigatioNN, self).__init__(observation_space, features_dim=features_dim)

        self.passthrough = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        shape = observations.shape
        observations = observations[:,0,:].reshape(shape[0], shape[-1])
        return self.passthrough(observations)

class PerceptionNavigationExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space) of dimension (1, 3, N_sensors)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    #def __init__(self, observation_space: gym.spaces.Dict, sensor_dim : int = 180, features_dim: int = 32, kernel_overlap : float = 0.05):
    def __init__(self, observation_space: gym.spaces.Dict, sensor_dim: int = 180, features_dim: int = 11, kernel_overlap: float = 0.25):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(PerceptionNavigationExtractor, self).__init__(observation_space, features_dim=1)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "perception":
                # Pass sensor readings through CNN
                extractors[key] = RadarCNN(subspace, sensor_dim=sensor_dim, features_dim=features_dim, kernel_overlap=kernel_overlap)
                total_concat_size += features_dim  # extractors[key].n_flatten
            elif key == "navigation":
                # Pass navigation features straight through to the MlpPolicy.
                extractors[key] = NavigatioNN(subspace, features_dim=subspace.shape[-1]) #nn.Identity()
                total_concat_size += subspace.shape[-1]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.ppo.policies import MlpPolicy
    from stable_baselines3 import PPO

    #### Test RadarCNN network circular 1D convolution:
    # Hyperparams
    n_sensors = 180
    kernel = 4
    padding = 4
    stride = 1

    ## Synthetic observation: (batch x channels x n_sensors)
    # Let obstacle be detected in the "edge" of the sensor array.
    # If circular padding works, should affect the outputs of the first <padding> elements
    obs = np.zeros((8, 3, n_sensors))
    obs[:, 0, :] = 150.0  # max distance
    obs[:, 0, -9:-1] = 10.0   # obstacle detected close in last 9 sensors
    obs[:, 1, :] = 0.0      # no obstacle
    obs[:, 2, :] = 0.0      # no obstacle
    obs = th.as_tensor(obs).float()

    ## Load existing convnet
    def load_net():
        from . import gym_auv
        algo = PPO
        #path = "radarCNN_example_Network.pkl"
        path = "../../radarCNN_example_Network150000.pkl"
        #path = "PPO_MlpPolicy_trained.pkl"
        #model = th.load(path)  # RunTimeError: : [enforce fail at ..\caffe2\serialize\inline_container.cc:114] . file in archive is not in a subdirectory: data
        #model = MlpPolicy.load(path)
        model = algo.load(path)


    load_net()
    print("loaded net")
    exit()
    ## Initialize convolutional layers (circular padding in all layers or just the first?)
    # First layer retains spatial structure,
    # includes circular padding to maintain the continuous radial structure of the RADAR,
    # and increased the feature-space dimensionality for extrapolation
    # (other padding types:)
    net1 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=kernel, padding=padding,
                      padding_mode='circular', stride=stride)
    # Second layer
    net2 = nn.Conv1d(in_channels=6, out_channels=3, kernel_size=kernel, padding=padding,
                      padding_mode='circular', stride=stride)
    net3 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=kernel, stride=2)
    net4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel, stride=2)

    flatten = nn.Flatten()
    act = nn.ReLU()
    #conv_weights = np.zeros(net1.weight.shape)

    #out1 = net1(obs)
    #out2 = net2(out1)
    #out3 = net3(out2)
    #out4 = net4(out3)
    out1 = act(net1(obs))
    out2 = act(net2(out1))
    out3 = act(net3(out2))
    out4 = act(net4(out3))

    feat = flatten(out4)


    ## Print shapes and characteritics of intermediate layer outputs
    obs = obs.detach().numpy()
    out1 = out1.detach().numpy()
    out2 = out2.detach().numpy()
    out3 = out3.detach().numpy()
    out4 = out4.detach().numpy()
    feat = feat.detach().numpy()

    def th2np_info(arr):
        #arr = tensor.detach().numpy()
        return "{:15.2f}{:15.2f}{:15.2f}{:15.2f}".format(arr.mean(), arr.std(), np.min(arr), np.max(arr))

    print("Observation",     obs.shape,  th2np_info(obs))
    print("First layer",     out1.shape, th2np_info(out1))
    print("Second layer",    out2.shape, th2np_info(out2))
    print("Third layer",     out3.shape, th2np_info(out3))
    print("Fourth layer",    out4.shape, th2np_info(out4))
    print("Output features", feat.shape, th2np_info(feat))

    ## PLOTTING
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    # plt.rc('font', family='serif', serif='Times')
    # plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)
    plt.axis('scaled')
    def feat2radar(feat, avg=False):
        # Find length of feature vector
        n = feat.shape[-1] # number of activations differ between conv-layers
        feat = np.mean(feat, axis=0) if avg else feat[0] # average activations over batch or just select one

        # Find angles for each feature
        theta_d = 2 * np.pi / n  # Spatial spread according to the number of actications
        theta = np.array([(i + 1)*theta_d for i in range(n)]) # Angles for each activation

        # Hotfix: append first element of each list to connect the ends of the lines in the plot.
        theta = np.append(theta, theta[0])
        if len(feat.shape) > 1:
            _feat = []
            for ch, f in enumerate(feat):
                ext = np.concatenate((f, [f[0]]))
                _feat.append(ext)
        else:
            _feat = np.append(feat, feat[0])

        _feat = np.array(_feat)
        return theta, _feat  # Return angle positions & list of features.

    # sensor angles : -pi -> pi, such that the first sensor is directly behind the vessel, and sensors go counter-clockwise around to the back again.
    _d_sensor_angle = 2 * np.pi / n_sensors
    sensor_angles = np.array([(i + 1)*_d_sensor_angle for i in range(n_sensors)])
    sensor_distances = obs[0,0,:]

    # hotfix to connect lines on start and end
    sensor_angles = np.append(sensor_angles, sensor_angles[0])
    sensor_distances = np.append(sensor_distances, sensor_distances[0])


    fig, ax = plt.subplots(figsize=(11,11), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("S")
    ax.plot(sensor_angles, sensor_distances)
    ax.set_rmax(1)
    ax.set_rticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # Less radial ticks
    ax.set_rlabel_position(22.5)  # Move radial labels away from plotted line
    #ax.set_rscale('symlog')
    ax.grid(True)

    ax.set_title("RadarCNN: intermediate layers visualization", va='bottom')

    to_plot = [obs, out1, out2, out3, out4, feat]
    names = ["obs", "out1", "out2", "out3", "out4", "feat"]
    channels = [0,1,2,3,4,5]
    linetypes = ["solid", 'dotted', 'dashed', 'dashdot', (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5))]
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']

    layer_color = {
        'obs' : '#377eb8',
        'out1': '#ff7f00',
        'out2': '#4daf4a',
        'out3': '#f781bf',
        'out4': '#a65628',
        'feat': '#984ea3',
    }

    for arr, layer in zip(to_plot, names):
        angle, data = feat2radar(arr, avg=False)
        if len(data.shape) > 1:
            for ch, _d in enumerate(data):
                ax.plot(angle, _d, linestyle=linetypes[ch], color=layer_color[layer], label=layer+'_ch'+str(ch))
        else:
            ax.plot(angle, data, linestyle=linetypes[0], color=layer_color[layer], label=layer)

    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
    plt.tight_layout()
    plt.show()
