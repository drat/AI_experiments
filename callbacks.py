from fastai.conv_learner import *

class SaveModel(Callback):
    """
    Callback that saves the model at the end of each cycle (phase).
    """
    
    def __init__(self, learn, save_name):
        self.learn, self.save_name = learn, save_name
    
    def on_train_begin(self):
        self.phase = 0
        
    def on_phase_end(self):
        if self.phase != 0: self.learn.save(self.save_name + str(self.phase))
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
        for group_param in self.learn.sched.layer_opt.opt_params():
            for param in group_param['params']:
                self.add_deciles(self.pnames[param], to_np(param))
                self.add_deciles(self.pnames[param] + '.grad', to_np(param.grad))
    
    def separate_pcts(self,arr):
        n = len(arr)
        pos, neg = arr[arr > 0], arr[arr < 0]
        pos_pcts = np.percentile(pos, self.pcts) if len(pos) > 0 else np.array([])
        neg_pcts = np.percentile(neg, self.pcts) if len(neg) > 0 else np.array([])
        return len(pos)/n, len(neg)/n, pos_pcts, neg_pcts
    
    def add_deciles(self, name, arr):
        pos, neg, pct_pos, pct_neg = self.separate_pcts(arr)
        self.deciles[name + 'sgn'].append([pos, neg])
        self.deciles[name + 'pos'].append(pct_pos)
        self.deciles[name + 'neg'].append(pct_neg)
                                                        
    def on_train_end(self):
        with open(self.fname, 'a') as f: f.write(self.logs)
        pickle.dump(self.deciles, open(self.fname[:-4] + '.pkl', 'wb'))