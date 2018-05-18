from fastai.conv_learner import *
import fire

class SaveModel(Callback):
    """
    Callback that saves the model at the end of each cycle (phase).
    """
    
    def __init__(self, learn, save_name):
        self.learn, self.save_name = learn, save_name
    
    def on_train_begin(self):
        self.phase = 0
        
    def on_phase_end(self):
        if self.phase != 0: learn.save(self.save_name + str(self.phase))
        self.phase += 1

class LogResults(Callback):
    """
    Callback to log all the results of the training:
    - at the end of each epoch: training loss, validation loss and metrics
    - at the end of the first batches then every epoch: deciles of the params and their gradients
    """
    
    def __init__(self, learn, fname, init_text):
        super().__init__()
        self.learn, self.fname, self.init_text = learn, fname, init_text
        self.pcts = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
        self.rec_first_batches = [1, 2, 4, 8, 16, 32, 64]
        self.pnames = {p:n for n,p in learn.model.named_parameters()}
        
    def on_train_begin(self):
        self.logs, self.epoch, self.n = self.init_text + "\n", 0, 0
        self.deciles = collections.defaultdict(list)
        names = ["epoch", "trn_loss", "val_loss", "metric"]
        layout = "{!s:10} " * len(names)
        self.logs += layout.format(*names) + "\n"
    
    def on_batch_end(self, metrics):
        self.loss = metrics
        if self.epoch == 0:
            self.n += 1
            if self.n in self.rec_first_batches:
                self.save_deciles()
    
    def on_epoch_end(self, metrics):
        self.save_stats(self.epoch, [self.loss] + metrics)
        self.epoch += 1
        self.save_deciles()
        
    def save_stats(self, epoch, values, decimals=6):
        layout = "{!s:^10}" + " {!s:10}" * len(values)
        values = [epoch] + list(np.round(values, decimals))
        self.logs += layout.format(*values) + "\n"
    
    def save_deciles(self):
        for group_param in self.sched.layer_opt.opt_params():
            for param in group_param['params']:
                self.add_deciles(self.pnames[param], to_np(param))
                self.add_deciles(self.pnames[param] + '.grad', to_np(param.grad))
    
    def separate_pcts(self,arr):
        n = len(arr)
        pos = arr[arr > 0]
        neg = arr[arr < 0]
        return pos/n, neg/n, np.percentile(pos, self.pcts), np.percentile(neg, self.pcts)
    
    def add_deciles(self, name, arr):
        pos, neg, pct_pos, pct_neg = self.separate_pcts(arr)
        self.deciles[name + 'sgn'].append([pos, neg])
        self.deciles[name + 'pos'].append(pct_pos)
        self.deciles[name + 'neg'].append(pct_neg)
                                                        
    def on_train_end(self):
        with open(self.fname, 'a') as f: f.write(self.logs)
        pickle.dump(self.deciles, open(self.fname[:-4] + '.pkl', 'wb'))

def CAR_phases(lr, n_cyc, moms, opt_fn, cyc_len, cyc_mul, wd, wd_loss):
    """
    Cosine annealing with restarts
    """
    if isinstance(moms, Iterable):
        phases = [TrainingPhase(cyc_len/ 20, opt_fn, lr=lr/100, momentum=moms[0], wds=wd, wd_loss=wd_loss),
                  TrainingPhase(cyc_len * 19/20, opt_fn, lr=lr, lr_decay=DecayType.COSINE, momentum=moms,
                                momentum_decay=DecayType.LINEAR, wds=wd, wd_loss=wd_loss)]
        for i in range(1,n_cyc):
            phases.append(TrainingPhase(cyc_len * (cyc_mul**i), opt_fn, lr=lr, lr_decay=DecayType.COSINE,
                                        momentum=moms, momentum_decay=DecayType.LINEAR, wds=wd, wd_loss=wd_loss))
    else:
        phases = [TrainingPhase(cyc_len/ 20, opt_fn, lr=lr/100, momentum=moms, wds=wd, wd_loss=wd_loss),
                  TrainingPhase(cyc_len * 19/20, opt_fn, lr=lr, lr_decay=DecayType.COSINE, momentum=moms, wds=wd, wd_loss=wd_loss)]
        for i in range(1,n_cyc):
            phases.append(TrainingPhase(cyc_len * (cyc_mul**i), opt_fn, lr=lr, lr_decay=DecayType.COSINE,
                                        momentum=moms, wds=wd, wd_loss=wd_loss))
    return phases

def CLR_phases(lr, n_cyc, moms, opt_fn, cyc_len, div, wd, wd_loss):
    """
    Cyclical learning rates
    """
    if isinstance(moms, Iterable):
        cyc_phases = [TrainingPhase(cyc_len/ 2, opt_fn, lr=(lr/div,lr), lr_decay=DecayType.LINEAR, momentum=moms, 
                                    momentum_decay=DecayType.LINEAR, wds=wd, wd_loss=wd_loss),
                      TrainingPhase(cyc_len/ 2, opt_fn, lr=(lr,lr/div), lr_decay=DecayType.LINEAR, momentum=(moms[1],moms[0]), 
                                    momentum_decay=DecayType.LINEAR, wds=wd, wd_loss=wd_loss)]
    else:
        cyc_phases = [TrainingPhase(cyc_len/ 2, opt_fn, lr=(lr/div,lr), lr_decay=DecayType.LINEAR, momentum=moms, 
                                    wds=wd, wd_loss=wd_loss),
                      TrainingPhase(cyc_len/ 2, opt_fn, lr=(lr,lr/div), lr_decay=DecayType.LINEAR, momentum=moms, 
                                    wds=wd, wd_loss=wd_loss)]
    return cyc_phases * n_cyc

def OCY_phases(lr, moms, opt_fn, cyc_len, div, pct, wd, wd_loss):
    """
    1cycle
    """
    phases = CLR_phases(lr, 1, moms, opt_fn, cyc_len * (1-pct), div, wd, wd_loss)
    mom = moms[0] if isinstance(moms, Iterable) else moms
    phases.append(TrainingPhase(pct * cyc_len, opt_fn, lr=(lr/div,lr/(100*div)), lr_decay=DecayType.LINEAR, momentum=mom, 
                                    wds=wd, wd_loss=wd_loss))
    return phases

def launch_dogscats(lr, mom, bs=64, mom2=None, wd=0, trn_met='CAR', n_cyc=1, cyc_len=1, cyc_mul=1, div=10, pct = 0.1, opt_fn='Adam', 
                   beta=None, wd_loss=True, snap=False, swa=False, tta=False, amsgrad=False, cuda_id=0, name=''):
    assert trn_met in {'CAR', 'CLR', '1CY'}, 'trn_met should be CAR (Cos Anneal with restart), CLR (cyclical learning rates) or 1CY (1cycle)'
    assert opt_fn in {'SGD', 'RMSProp', 'Adam'}, 'optim should be SGD, RMSProp or Adam'
    torch.cuda.set_device(cuda_id)
    init_text = f'{name}_{cuda_id}\n'
    init_text = f'lr {lr}; max_mom {mom}; batch_size {bs}; min_mom {mom2}; weight_decay {wd} train_method {trn_met}; num_cycles {n_cyc}; '
    init_text += f'cycle_len {cyc_len}; cycle_mult {cyc_mul}; lr_div {div}; pct_relax {pct}; optimizer {opt_fn}; beta {beta}; ' 
    init_text += f'wd_in_loss {wd_loss}; snapshot_ensemble {snap}; use_swa {swa}; tta {tta}; amsgrad {amsgrad}'
    print(init_text)
    PATH = Path("../data/dogscats/")
    sz=224
    arch=resnet34
    data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz), bs=bs)
    learn = ConvLearner.pretrained(arch, data, precompute=True)
    if opt_fn=='SGD': opt_fn = optim.SGD
    elif opt_fn=='RMSProp': opt_fn = optim.RMSprop if beta is None else partial(optim.RMSProp, alpha=beta)
    else: opt_fn = partial(optim.Adam, amsgrad=amsgrad) if beta is None else partial(optim.Adam, betas=(mom,beta), amsgrad=amsgrad)
    learn.opt_fn = opt_fn
    moms = mom if mom2 is None else (mom,mom2)
    learn.precompute=False
    if not isinstance(n_cyc,tuple): n_cyc = (n_cyc,n_cyc)
    if not isinstance(cyc_len,tuple): cyc_len = (cyc_len,cyc_len)
    if not isinstance(cyc_mul,tuple): cyc_mul = (cyc_mul,cyc_mul)
    if trn_met=='CAR': trn_phases = CAR_phases(lr, n_cyc[0], moms, opt_fn, cyc_len[0], cyc_mul[0], wd, wd_loss)
    elif trn_met=='CLR': trn_phases = CLR_phases(lr, n_cyc[0], moms, opt_fn, cyc_len[0], div, wd, wd_loss)
    else: trn_phases = OCY_phases(lr, moms, opt_fn, cyc_len[0], div, pct, wd, wd_loss)
    log_rec = LogResults(f'logs_{name}_{cuda_id}.txt', init_text + '\n\nPhase1')
    learn.fit_opt_sched(trn_phases, use_swa=swa, callbacks=[log_rec])
    learn.unfreeze()
    lrs = np.array([lr/100, lr/10, lr])
    if trn_met=='CAR': trn_phases = CAR_phases(lrs, n_cyc[1], moms, opt_fn, cyc_len[1], cyc_mul[1], wd, wd_loss)
    elif trn_met=='CLR': trn_phases = CLR_phases(lrs, n_cyc[1], moms, opt_fn, cyc_len[1], div, wd, wd_loss)
    else: trn_phases = OCY_phases(lrs, moms, opt_fn, cyc_len[1], div, pct, wd, wd_loss)
    log_rec = LogResults(f'logs_{name}_{cuda_id}.txt', '\nPhase2')
    cbs = [log_rec]
    if snap: cbs.append(SaveModel(learn, 'cycle'))
    learn.fit_opt_sched(trn_phases, use_swa=swa, callbacks=cbs)
    if tta or snap: 
        probs, targs = get_probs(learn, n_cyc[1], tta, snap)
        acc = accuracy_np(probs, targs)
        loss = F.nll_loss(V(np.log(probs)), V(targs)).item()
        with open(f'logs_{name}_{cuda_id}.txt','a') as f: f.write('\n' + f'Final loss: {loss}     Final accuracy: {acc}')
        
def get_probs(learn, tta, snap):
    if tta and not snap: return learn.TTA()
    probs = []
    for i in range(1,n_cyc+1):
        learn.load('cycle' + str(i))
        preds, targs = learn.predict_with_targs() if not tta else learn.TTA()
        probs.append(np.exp(preds)[None])
    probs = np.concatenate(probs, 0)
    return np.mean(probs, 0), targs
            
def train_dogscats(lr, mom, bs=64, mom2=None, wd=0, trn_met='CAR', n_cyc=1, cyc_len=1, cyc_mul=1, div=10, pct = 0.1, opt_fn='Adam', 
                   beta=None, wd_loss=True, snap=False, swa=False, tta=False, amsgrad=False, cuda_id=0, name=''):
    if os.path.isfile(f'logs_{name}_{cuda_id}.txt'):
        os.remove(f'logs_{name}_{cuda_id}.txt')
    for i in range(0,5):
        launch_dogscats(lr, mom, bs, mom2, wd, trn_met, n_cyc, cyc_len, cyc_mul, div, pct, opt_fn, 
                   beta, wd_loss, snap, swa, tta, amsgrad, cuda_id, name)
    
if __name__ == '__main__': fire.Fire(train_dogscats)

