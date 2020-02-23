from nips2018.movie.parameters import DataConfig
from nips2018.architectures.readouts import ST3dSharedGridStopGradientReadout
from nips2018.architectures.cores import StackedFeatureGRUCore
from nips2018.architectures.shifters import StaticAffineShifter
from nips2018.architectures.modulators import GateGRUModulator
from nips2018.architectures.base import CorePlusReadout3d

from attorch.layers import Elu1
from collections import OrderedDict

import torch


def load_checkpoint(model, filename):
    statedict = torch.load(filename)
    model.load_state_dict(statedict)
    

def load_model(groupid, filename):
    # get dataloader because number of neurons and img_shape are needed for loading the model
    test_batch = 1
    key = dict(data_hash='5253599d3dceed531841271d6eeba9c5', group_id=groupid, seed=2606)
    testsets, testloaders = DataConfig().load_data(key, tier='test', batch_size=test_batch)
    n_neurons = OrderedDict([(k, v.n_neurons) for k, v in testsets.items()])
    img_shape = list(testloaders.values())[0].dataset.img_shape

    # load the model
    core = StackedFeatureGRUCore(
        input_channels=1,
        hidden_channels=12,
        rec_channels=36,
        input_kern=7,
        hidden_kern=3, 
        rec_kern=3, 
        layers=3,
        gamma_input=50, 
        gamma_hidden=.1, 
        gamma_rec=.0, 
        momentum=.1,
        skip=2, 
        bias=False, 
        batch_norm=True, 
        pad_input=True)

    ro_in_shape = CorePlusReadout3d.get_readout_in_shape(core, img_shape)

    readout = ST3dSharedGridStopGradientReadout(
        ro_in_shape,
        n_neurons,
        positive=False,
        gamma_features=1.,
        pool_steps=2,
        kernel_size=4,
        stride=4,
        gradient_pass_mod=3)

    shifter = StaticAffineShifter(
        n_neurons, 
        input_channels=2, 
        hidden_channels=2, 
        bias=True, 
        gamma_shifter=0.001)

    modulator = GateGRUModulator(
        n_neurons, 
        gamma_modulator=0.0, 
        hidden_channels=50, 
        offset=1, 
        bias=True)

    model = CorePlusReadout3d(
        core, 
        readout, 
        nonlinearity=Elu1(),
        shifter=shifter, 
        modulator=modulator, 
        burn_in=15)

    load_checkpoint(model, filename)
    return model