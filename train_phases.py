from fastai.conv_learner import *

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
