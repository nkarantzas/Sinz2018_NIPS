import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import style
from nips2018.movie.parameters import DataConfig

style.use('ggplot')


def corr(y1, y2, axis=-1, eps=1e-8):
    y1 = (y1 - y1.mean(axis=axis))/(y1.std(axis=axis) + eps)
    y2 = (y2 - y2.mean(axis=axis))/(y2.std(axis=axis) + eps)
    return (y1 * y2).mean(axis=axis)


def test_correlations(testsets, testloaders, model, constant_eye=None, constant_beh=None):
    for readout_key, loader in testloaders.items():
        y = []
        y_hat = []
        for inputs, behavior, eye, responses in loader:
            neurons = responses.size(-1) 
            outputs = model(
                inputs.cuda(), 
                readout_key=readout_key, 
                eye_pos=constant_eye.cuda() if constant_eye is not None else eye.cuda(),
                behavior=constant_beh.cuda() if constant_beh is not None else behavior.cuda()).data.cpu().numpy()
            lag = responses.shape[1] - outputs.shape[1]
            responses = responses[:, lag:, :].data.cpu().numpy().reshape((-1, neurons))
            outputs = outputs.reshape((-1, neurons))
            y.append(responses)
            y_hat.append(outputs)
        test_pearsons = corr(np.vstack(y), np.vstack(y_hat), axis=0)
        unit_ids = testsets[readout_key].neurons.unit_ids
        Dict = [dict(pearson=c, unit_id=u) for u, c in zip(unit_ids, test_pearsons)]
    return test_pearsons


def oracle_correlations(key):
    testsets, testloaders = DataConfig().load_data(key, tier='test', oracle=True)
    for readout_key, loader in testloaders.items():
        oracles, outs = [], []
        for *_, responses in loader:
            # oracle responses. These are in chunks based on the repetitions
            responses = responses.numpy()
            
            # mean per neuron over the clip repetitions
            mu = responses.mean(axis=0, keepdims=True)
            
            # leave one out
            repeats = responses.shape[0]
            oracle = (mu*repeats - responses) / (repeats - 1)
            
            oracles.append(oracle.reshape(-1, responses.shape[-1]))
            outs.append(responses.reshape(-1, responses.shape[-1]))

        oracle_pearsons = corr(np.vstack(outs), np.vstack(oracles), axis=0)
        unit_ids = testsets[readout_key].neurons.unit_ids
        Dict = [dict(pearson=c, unit_id=u) for u, c in zip(unit_ids, oracle_pearsons)]
    return oracle_pearsons


def mean_eye_beh(testloaders):
    for readout_key, testloader in testloaders.items():
        loader = testloaders[readout_key]
        eyes = []
        behs = []
        for _, behavior, eye, _ in loader:
            nframes = eye.shape[1]
            
            eye = eye.view(-1, eye.shape[-1])
            eyes.append(eye)
            
            behavior = behavior.view(-1, behavior.shape[-1])
            behs.append(behavior)

        eyes = torch.cat(eyes, 0)
        eyes = eyes.mean(0)

        behs = torch.cat(behs, 0)
        behs = behs.mean(0)
        
    eyes = eyes.expand(nframes, 2).resize(1, nframes, 2)
    behs = behs.expand(nframes, 3).resize(1, nframes, 3)
    return eyes, behs
    

def fraction_oracle(oracle, test, savepath=None, title=None):
    oracle = oracle[:, np.newaxis]
    fraction, _, _, _ = np.linalg.lstsq(oracle, test)
    plt.figure(figsize=(8, 8))
    plt.scatter(oracle, test, color='dodgerblue', alpha=0.5)
    plt.plot([0, .7], [0, .7], '--', color='darkslategray')
    plt.plot(
        oracle, 
        fraction*oracle, 
        linewidth=5, 
        color='dodgerblue', 
        alpha=0.5,
        label='{:.1f}% oracle '.format(fraction.item() * 100))
    
    plt.xlabel('Oracle Correlation', fontsize=16)
    plt.ylabel('Test Correlation', fontsize=16)
    if title is not None:
        plt.title(title)
    plt.xlim([-.1, .9])
    plt.ylim([-.1, .9])
    plt.legend(fontsize=14)
    if savepath:
        plt.savefig(savepath)
    plt.show()