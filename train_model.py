from fastai.conv_learner import *
from train_phases import *
from callbacks import *
from sklearn.metrics import fbeta_score
import fire

def f2(preds, targs, start=0.17, end=0.24, step=0.01):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return max([fbeta_score(targs, (preds>th), 2, average='samples')
                    for th in np.arange(start,end,step)])

def get_data(data, bs):
    PATH = Path(f'../data/{data}/')
    if data=='dogscats':
        sz, arch = 224, resnet34
        tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.05)
        data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)
        learn = ConvLearner.pretrained(arch, data)
        frozen, log_probs = True, True
    elif data=='planet':
        sz, arch = 256, resnet34
        label_csv = PATH/'train_v2.csv'
        n = len(list(open(label_csv)))-1
        val_idxs = get_cv_idxs(n)
        tfms = tfms_from_model(arch, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
        data = ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms,
                    suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg')
        data = data.resize(int(sz*1.3), 'tmp')
        learn = ConvLearner.pretrained(arch, data, metrics=[f2])
        frozen, log_probs = True, False
    elif data=='cifar10':
        pass
    return learn, frozen, log_probs

def get_opt_fn(opt_fn, mom, beta, amsgrad):
    if opt_fn=='SGD': res = optim.SGD
    elif opt_fn=='RMSProp': res = optim.RMSprop if beta is None else partial(optim.RMSProp, alpha=beta)
    else: res = partial(optim.Adam, amsgrad=amsgrad) if beta is None else partial(optim.Adam, betas=(mom,beta), amsgrad=amsgrad)
    return res

def get_trn_phases(trn_met, lr, n_cyc, moms, opt_fn, cyc_len, cyc_mul, div, pct, wd, wd_loss):
    if trn_met=='CAR': trn_phases = CAR_phases(lr, n_cyc, moms, opt_fn, cyc_len, cyc_mul, wd, wd_loss)
    elif trn_met=='CLR': trn_phases = CLR_phases(lr, n_cyc, moms, opt_fn, cyc_len, div, wd, wd_loss)
    else: trn_phases = OCY_phases(lr, moms, opt_fn, cyc_len, div, pct, wd, wd_loss)
    return trn_phases

def launch_training(lr, mom, bs=64, mom2=None, wd=0, trn_met='CAR', n_cyc=1, cyc_len=1, cyc_mul=1, div=10, pct = 0.1, 
                    opt_fn='Adam', beta=None, wd_loss=True, snap=False, swa=False, tta=False, amsgrad=False, cuda_id=0, name='',
                    data='dogscats', freeze_first=None, div_diff_lr=None):
    assert trn_met in {'CAR', 'CLR', '1CY'}, 'trn_met should be CAR (Cos Anneal with restart), CLR (cyclical learning rates) or 1CY (1cycle)'
    assert opt_fn in {'SGD', 'RMSProp', 'Adam'}, 'optim should be SGD, RMSProp or Adam'
    torch.cuda.set_device(cuda_id)
    init_text = f'{name}_{cuda_id}\n'
    init_text = f'lr {lr}; max_mom {mom}; batch_size {bs}; min_mom {mom2}; weight_decay {wd} train_method {trn_met}; num_cycles {n_cyc}; '
    init_text += f'cycle_len {cyc_len}; cycle_mult {cyc_mul}; lr_div {div}; pct_relax {pct}; optimizer {opt_fn}; beta {beta}; ' 
    init_text += f'wd_in_loss {wd_loss}; snapshot_ensemble {snap}; use_swa {swa}; tta {tta}; amsgrad {amsgrad}; data {data}; '
    init_text += f'freeze_first {freeze_first}'
    print(init_text)
    learn, frozen, log_probs = get_data(data, bs)
    opt_fn = get_opt_fn(opt_fn, mom, beta, amsgrad)
    learn.opt_fn = opt_fn
    moms = mom if mom2 is None else (mom,mom2)
    if freeze_first is None: freeze_first=frozen
    if freeze_first:
        if not isinstance(lr, tuple): lr = (lr, lr)
        if not isinstance(n_cyc,tuple): n_cyc = (n_cyc,n_cyc)
        if not isinstance(cyc_len,tuple): cyc_len = (cyc_len,cyc_len)
        if not isinstance(cyc_mul,tuple): cyc_mul = (cyc_mul,cyc_mul)
        trn_phases = get_trn_phases(trn_met, lr[0], n_cyc[0], moms, opt_fn, cyc_len[0], cyc_mul[0], div, pct, wd, wd_loss)
        log_rec = LogResults(learn, f'logs_{name}_{cuda_id}.txt', init_text + '\n\nPhase1')
        learn.fit_opt_sched(trn_phases, use_swa=swa, callbacks=[log_rec])
        learn.unfreeze()
        lr, n_cyc, cyc_len, cyc_mul = lr[1], n_cyc[1], cyc_len[1], cyc_mul[1]
    if div_diff_lr is None: 
        div_diff_lr = 10 if data=='dogscats' else (3 if data=='planets' else 1)
    if div_diff_lr != 1: lrs = np.array([lr/(div_diff_lr**2), lr/div_diff_lr, lr])
    else: lrs = lr
    trn_phases = get_trn_phases(trn_met, lrs, n_cyc, moms, opt_fn, cyc_len, cyc_mul, div, pct, wd, wd_loss)
    log_rec = LogResults(learn, f'logs_{name}_{cuda_id}.txt', '\nPhase2')
    cbs = [log_rec]
    if snap: cbs.append(SaveModel(learn, 'cycle'))
    learn.fit_opt_sched(trn_phases, use_swa=swa, callbacks=cbs)
    if tta or snap:
        probs, targs = get_probs(learn, n_cyc, tta, snap, log_probs)
        acc = learn.metrics[0](V(probs), V(targs))
        if log_probs: probs = np.log(probs)
        loss = learn.crit(V(probs), V(targs)).item()
        print(f'Final loss: {loss}     Final metric: {acc}')
        with open(f'logs_{name}_{cuda_id}.txt','a') as f: f.write('\n' + f'Final loss: {loss}     Final metric: {acc}')
        
def get_probs(learn, n_cyc, tta, snap, logs):
    if tta and not snap: 
        preds, targs = learn.TTA()
        if logs: probs = np.exp(preds)
        return np.mean(probs,0), targs
    probs = []
    for i in range(1,n_cyc+1):
        learn.load('cycle' + str(i))
        preds, targs = learn.predict_with_targs() if not tta else learn.TTA()
        if logs: preds = np.exp(preds)
        if tta: preds = np.mean(preds,0)
        probs.append(preds[None])
    probs = np.concatenate(probs, 0)
    return np.mean(probs, 0), targs
            
def train_model(lr, mom, bs=64, mom2=None, wd=0, trn_met='CAR', n_cyc=1, cyc_len=1, cyc_mul=1, div=10, pct = 0.1, opt_fn='Adam', 
                beta=None, wd_loss=True, snap=False, swa=False, tta=False, amsgrad=False, cuda_id=0, name='', data='dogscats',
                freeze_first=None, div_diff_lr=None, nb_exp=5):
    if os.path.isfile(f'logs_{name}_{cuda_id}.txt'):
        os.remove(f'logs_{name}_{cuda_id}.txt')
    for i in range(0,nb_exp):
        launch_training(lr, mom, bs, mom2, wd, trn_met, n_cyc, cyc_len, cyc_mul, div, pct, opt_fn, 
                   beta, wd_loss, snap, swa, tta, amsgrad, cuda_id, name, data, freeze_first, div_diff_lr)
    
if __name__ == '__main__': fire.Fire(train_model)

