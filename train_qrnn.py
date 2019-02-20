from fastai.text import * 
import fastprogress
fastprogress.fastprogress.SAVE_PATH = 'fastec2/spot2/log.txt'

path = Config().data_path()/'wikitext-2'

def istitle(line):
    return len(re.findall(r'^ = [^=]* = $', line)) != 0

def read_file(filename):
    articles = []
    with open(filename, encoding='utf8') as f:
        lines = f.readlines()
    current_article = ''
    for i,line in enumerate(lines):
        current_article += line
        if i < len(lines)-2 and lines[i+1] == ' \n' and istitle(lines[i+2]):
            current_article = current_article.replace('<unk>', UNK)
            articles.append(current_article)
            current_article = ''
    current_article = current_article.replace('<unk>', UNK)
    articles.append(current_article)
    return np.array(articles)

train = read_file(path/'train.txt')
valid = read_file(path/'valid.txt')
test =  read_file(path/'test.txt')

all_texts = np.concatenate([valid, train])
df = pd.DataFrame({'texts':all_texts})
df_tst = pd.DataFrame({'texts':test})

df['texts'] = df['texts'].apply(lambda x:[BOS] + x.split(' '))
    
processor = [NumericalizeProcessor(min_freq=0)]
data = (TextList.from_df(df, path, cols='texts', processor=processor)
                .split_by_idx(range(0,60))
                .label_for_lm()
                .databunch(bs=100, bptt=70))

config = awd_lstm_lm_config.copy()
config['input_p']  = 0.4
config['output_p'] = 0.4
config['weight_p'] = 0.1
config['embed_p']  = 0.1
config['hidden_p'] = 0.2
config['qrnn'] = True

learn = language_model_learner(data, AWD_LSTM, config=config, pretrained=False, clip=0.1, alpha=2, beta=1)
learn.fit_one_cycle(90,5e-3,wd=0.1)

class Perplexity(Callback):
    
    def on_epoch_begin(self, **kwargs):
        self.loss,self.len = 0.,0
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        self.loss += last_target.size(1) * CrossEntropyFlat()(last_output, last_target)
        self.len += last_target.size(1)
    
    def on_epoch_end(self, **kwargs):
        self.metric = torch.exp(self.loss / self.len)
        
learn.metrics.append(Perplexity())

print("Validation set")
learn.validate()

print("Test set")
all_texts2 = np.concatenate([test, train])
df2 = pd.DataFrame({'texts':all_texts})

df2['texts'] = df2['texts'].apply(lambda x:[BOS] + x.split(' '))
    
processor = [NumericalizeProcessor(vocab = data.vocab)]
data2 = (TextList.from_df(df2, path, cols='texts', processor=processor)
                 .split_by_idx(range(0,60))
                 .label_for_lm()
                 .databunch(bs=100, bptt=70))
learn.data = data2
learn.validate()
