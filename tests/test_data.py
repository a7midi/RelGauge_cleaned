import json, os, numpy as np

def test_frozen_transition_and_support_present():
    assert os.path.exists('data/transition_iter8_inst28_seed3221741.npy')
    T=np.load('data/transition_iter8_inst28_seed3221741.npy')
    assert T.ndim==1 and len(T)==512
    ref=json.load(open('data/reference_support.json'))
    assert ref['support_size']==17
    assert ref['support_indices'][-1]==287
