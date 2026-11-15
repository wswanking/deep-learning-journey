"""Fixed-weight, float64 CPU forward propagation; no training takes place."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import platform
import time
STARTED=time.perf_counter()
import numpy as np
import torch
from figures import write_figures
ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'build/experiments/07-multilayer-perceptron/results.json')
    parser.add_argument('--skip-figures',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)  # Constructor initialization is entirely overwritten below.
    started=time.perf_counter()
    X=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float64)
    targets=np.array([0,1,1,0],dtype=np.int64)
    W1=np.array([[1,1],[1,1]],dtype=np.float64)
    b1=np.array([0,-1],dtype=np.float64)
    W2=np.array([[1,-2]],dtype=np.float64)
    b2=np.array([-.5],dtype=np.float64)
    Z1=X@W1.T+b1
    H=np.maximum(Z1,0)
    logits=H@W2.T+b2
    probabilities=1/(1+np.exp(-logits))
    predictions=(logits[:,0]>=0).astype(np.int64)
    expected_h=np.array([[0,0],[1,0],[1,0],[2,1]],dtype=np.float64)
    np.testing.assert_array_equal(H,expected_h)
    np.testing.assert_array_equal(logits[:,0],[-.5,.5,.5,-.5])
    np.testing.assert_array_equal(predictions,targets)

    net=torch.nn.Sequential(torch.nn.Linear(2,2,dtype=torch.float64),torch.nn.ReLU(),torch.nn.Linear(2,1,dtype=torch.float64))
    with torch.no_grad():
        net[0].weight.copy_(torch.from_numpy(W1));net[0].bias.copy_(torch.from_numpy(b1))
        net[2].weight.copy_(torch.from_numpy(W2));net[2].bias.copy_(torch.from_numpy(b2))
        tensor=torch.from_numpy(X)
        module_logits=net(tensor)
        module_probabilities=torch.sigmoid(module_logits)
        singles=torch.stack([net(row) for row in tensor])
        perm=torch.tensor([3,0,2,1])
        permuted=net(tensor[perm])
    np.testing.assert_allclose(module_logits.numpy(),logits,rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(module_probabilities.numpy(),probabilities,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(singles,module_logits,rtol=0,atol=0)
    torch.testing.assert_close(permuted,module_logits[perm],rtol=0,atol=0)
    parameter_count=sum(p.numel() for p in net.parameters())
    assert parameter_count==9

    effective_W=W2@W1
    effective_b=W2@b1+b2
    no_activation=(X@W1.T+b1)@W2.T+b2
    collapsed=X@effective_W.T+effective_b
    np.testing.assert_allclose(no_activation,collapsed,rtol=1e-13,atol=1e-13)
    collapsed_predictions=(collapsed[:,0]>=0).astype(np.int64)
    assert np.mean(collapsed_predictions==targets)==.75
    # A separate nondegenerate affine map checks the algebra beyond the XOR weights.
    A=np.array([[1,-2],[3,.5]],dtype=np.float64);a=np.array([.2,-.3])
    B=np.array([[.4,-1.1]],dtype=np.float64);b=np.array([.7])
    np.testing.assert_allclose((X@A.T+a)@B.T+b,X@(B@A).T+(B@a+b),rtol=1e-13,atol=1e-13)

    bias_logits=H@W2.T+0.25
    bias_predictions=(bias_logits[:,0]>=0).astype(np.int64)
    np.testing.assert_array_equal(bias_predictions,[1,1,1,1])
    assert np.mean(bias_predictions==targets)==0.5
    probes=np.array([-4,-1,0,1,4],dtype=np.float64)
    tprobes=torch.from_numpy(probes)
    np.testing.assert_allclose(torch.relu(tprobes).numpy(),np.maximum(probes,0),rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(torch.sigmoid(tprobes).numpy(),1/(1+np.exp(-probes)),rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(torch.tanh(tprobes).numpy(),np.tanh(probes),rtol=1e-13,atol=1e-13)
    report={'chapter':'07','passed':True,'run_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
                       'os':platform.platform(),'device':'cpu','dtype':'float64','torch_threads':torch.get_num_threads()},
        'configuration':{'constructor_seed':7,'all_weights_overwritten':True,'training_steps':0,
                         'rtol':1e-13,'atol':1e-13,'data':'All four entries of the XOR truth table; no held-out evaluation.'},
        'inputs':X.tolist(),'targets':targets.tolist(),'W1':W1.tolist(),'b1':b1.tolist(),'W2':W2.tolist(),'b2':b2.tolist(),
        'preactivation':Z1.tolist(),'hidden':H.tolist(),'logits':logits.tolist(),'probabilities':probabilities.tolist(),
        'predictions':predictions.tolist(),'truth_table_accuracy':float(np.mean(predictions==targets)),
        'parameter_count':parameter_count,'shapes':{'input':list(X.shape),'preactivation':list(Z1.shape),'hidden':list(H.shape),'logits':list(logits.shape)},
        'without_activation':{'effective_W':effective_W.tolist(),'effective_b':effective_b.tolist(),
                             'logits':collapsed.tolist(),'predictions':collapsed_predictions.tolist(),
                             'truth_table_accuracy':float(np.mean(collapsed_predictions==targets))},
        'output_bias_perturbation':{'bias':0.25,'logits':bias_logits.tolist(),'predictions':bias_predictions.tolist(),'truth_table_accuracy':0.5},
        'resources':{'calculation_seconds':time.perf_counter()-started,'including_imports_seconds':time.perf_counter()-STARTED,
                     'input_array_bytes':X.nbytes,'memory_scope':'Array payload only; process peak memory not measured. SVG generation excluded from timing.'}}
    if not args.skip_figures:write_figures(report,ROOT/'docs/assets/07-multilayer-perceptron')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('hidden=',H.tolist());print('logits=',logits[:,0].tolist());print('probabilities=',probabilities[:,0].tolist())
    print('parameters=9; XOR truth-table accuracy=1.0; identity-hidden comparison=0.75')
    print('PASS; report='+str(args.output.resolve()))


if __name__=='__main__':main()
