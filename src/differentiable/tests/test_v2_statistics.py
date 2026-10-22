"""Aggregation and split-isolation contracts for scientific reports."""
import copy
import numpy as np
import pytest
from src.differentiable.v2_evaluate import aggregate,load_config,DEFAULT_CONFIG


def test_paired_aggregation_excludes_constant_ratios_and_counts_failures():
    config=load_config(DEFAULT_CONFIG)
    config.update(physics=['scalar'],modes=['soft'],strengths=[0,.25],bootstrap_samples=100)
    rows=[]
    for strength in config['strengths']:
        for family,n,base in [('constant',64,1e-16),('square',64,1.),('square',128,2.)]:
            for j in range(2):
                value=base*(1. if strength==0 else .5)
                rows.append(dict(case_id=f'{family}-{n}-{j}',physics='scalar',mode='soft',strength=strength,
                     family=family,cells=n,valid=True,l1=value,l2=value,linf=value,
                     conservation_residual=[0.],maximum_effective_cfl=.2))
    report=aggregate(rows,config)['summary'][-1]
    assert report['macro_relative_l1']==.5
    assert report['trajectory_weighted_relative_l1']==.5
    np.testing.assert_allclose(report['macro_ci95'],[.5,.5])
    assert report['groups'][0]['relative_l1'] is None
    rows[-1]['valid']=False
    assert aggregate(rows,config)['summary'][-1]['failures']==1


def test_original_test_split_is_rejected(tmp_path):
    import json
    config=load_config(DEFAULT_CONFIG);config['split']='test'
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='validation only'):load_config(path)
