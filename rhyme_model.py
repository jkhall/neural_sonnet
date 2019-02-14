import torch
import torch.nn as nn
import torch.nn.functional as F
import util


class RhymeModel(nn.Module):
  def __init__(self, batch_size):
    super().__init__()
    self.char2idx, self.idx2char, self.max_word_len = util.get_char_dicts()
    self.emb_sz = 100
    self.hidden_sz = 50
    self.vocab_sz = len(self.idx2char)
    self.seq_len = self.max_word_len
    self.num_layers = 1
    self.num_dirs = 1
    self.bs = batch_size
    self.char_emb = nn.Embedding(self.vocab_sz, self.emb_sz)
    self.char_lstm = nn.LSTM(self.emb_sz, self.hidden_sz, num_layers = self.num_layers) # 1 layer by default
    self.hidden =""
    
  # seq_len, batch_sz, hidden_sz
  def init_hidden(self):
    h = torch.zeros(self.num_layers * self.num_dirs, self.bs, self.hidden_sz)
    if torch.cuda.is_available():
      h = h.cuda()
    return h
  
  def init_cs(self):
    cs = torch.zeros(self.num_layers * self.num_dirs, self.bs, self.hidden_sz)
    if torch.cuda.is_available():
      cs = cs.cuda()
    return cs
  
  # target and refs are hidden states obtained after feeding the last
  # character through the LSTM
  def loss_fn(self, target, refs):
    margin = 0.4 # I have no idea what this ought to be
    cos = torch.nn.CosineSimilarity(dim=2, eps=0.1)
    cs1 = cos(target, refs[0]).squeeze(0)
    cs2 = cos(target, refs[1]).squeeze(0)
    cs3 = cos(target, refs[2]).squeeze(0)
      
    # for every batch you have to sort the cos sims
    losses = []
    for i in range(self.bs):
      Q = sorted([cs1[i], cs2[i], cs3[i]], reverse=True)
      losses.append(max(0, margin - Q[0] + Q[1]))
      
    # should losses for the batches be summed?
    return sum(losses), (cs1, cs2, cs3)
  
  def feed_chars(self, emb):
    self.bs = emb.shape[0]
    h = self.init_hidden()
    c = self.init_cs()
    emb_t = emb.transpose(0, 1) # seq_len, bs, emb_dim

    out, (h, c) = self.char_lstm(emb_t, (h, c))
    
    return h

  def get_vector_rep(self, w):
    w_emb = self.char_emb(w)
    w_hidden = self.feed_chars(w_emb)
    
    return w_hidden
  
  def get_cos_sim(self, w1, w2):
    
    w1_emb = self.char_emb(w1)
    w2_emb = self.char_emb(w2)
    w1_hidden = self.feed_chars(w1_emb)
    w2_hidden = self.feed_chars(w2_emb)
    
    cos = torch.nn.CosineSimilarity(dim=2, eps=0.1)
    
    return cos(w1_hidden, w2_hidden)
  
  # feed chars through lstm until last, then return target states and ref states
  def forward(self, rhyme_data):
    # target is a list of indices
    target = [rd[0] for rd in rhyme_data] # at the word level
    refs = [torch.stack(rd[1]) for rd in rhyme_data]
    
    target = torch.stack(target)
    refs = torch.stack(refs)
    
    self.bs = len(rhyme_data)
    
    # embed target
    tar_embedding = self.char_emb(target)
    
    # embed refs
    r1_embedding = self.char_emb(refs[:, 0])
    r2_embedding = self.char_emb(refs[:, 1])
    r3_embedding = self.char_emb(refs[:, 2])
     
    # feed through lstm
    t_hidden = self.feed_chars(tar_embedding)
    r1_hidden = self.feed_chars(r1_embedding)
    r2_hidden = self.feed_chars(r2_embedding)
    r3_hidden = self.feed_chars(r3_embedding)
    
    
    loss, cos_sims = self.loss_fn(t_hidden, (r1_hidden, r2_hidden, r3_hidden))
    
    return loss, cos_sims


def train_rhyme_model(model, dataset, optimizer):
  for i in range(len(dataset) // bs):
    sample = dataset[i * bs: i*bs + bs]
    loss, _ = rm(sample)

    if type(loss) == int:
      losses.append(loss)
    else:
      losses.append(loss.item())
      loss.backward()
    
    optimizer.step()
    
    optimizer.zero_grad()
    
  return losses
