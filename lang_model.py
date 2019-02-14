import fastai
from fastai.text import *

#data_lm = TextLMDataBunch.from_folder("data")
#learn = language_model_learner(data_lm, text.models.awd_lstm, drop_mult=0.5)
#learn.load(r"awd-v1")

def build_model():
  data_lm = TextLMDataBunch.from_folder("data")
  learn = language_model_learner(data_lm, fastai.text.models.awd_lstm, drop_mult=0.5)
  learn.load(r"awd-v1")

  return learn

def generate_quatrain(seed, lm):
  if lm != None:
    s = lm.predict(seed, n_words=40)

    while s.count("eos") < 4:
      next_word = re.sub(re.escape(s), "", lm.predict(s, n_words=1))
      s += next_word
      
    return s






def enforce_rhyme(lines, scheme, rhyme_index, rhyme_w, rhyme_v):
#   lines = poem.split("\n")
#   lines = list(map(lambda l: l.strip(), lines))
  if scheme == "ABBA":
    A = lines[0].split()[-1]
#     A_candidates = pronouncing.rhymes(A[-2:])
    B = lines[1].split()[-1]
#     B_candidates = pronouncing.rhymes(B[-2:])
    if len(np.where(rhyme_w==A)[0]) <= 0:
      # return none
      return None
    if len(np.where(rhyme_w==B)[0]) <= 0:
      return None
    a_ind = np.where(rhyme_w==A)[0][0]
    b_ind = np.where(rhyme_w==B)[0][0]
  
#     a_ind = rhyme_w.index(A)
#     b_ind = rhyme_w.index(B)
    
    a_vec = rhyme_v[a_ind]
    b_vec = rhyme_v[b_ind]
    
    a_rhyme_idxs, a_dists = rhyme_index.knnQuery(a_vec, k=5)
    b_rhyme_idxs, b_dists = rhyme_index.knnQuery(b_vec, k=5)
    
#     print("A rhymes: {}".format(rhyme_w[a_rhyme_idxs]))
#     print("B rhymes: {}".format(rhyme_w[b_rhyme_idxs]))
    
  
    A_candidates = rhyme_w[a_rhyme_idxs]
    B_candidates = rhyme_w[b_rhyme_idxs]
#     print("A ending: {}".format(A[-2:]))
#     print("B ending: {}".format(B[-2:]))
    
#     # make replacements
    a_idx = math.floor(random.random() * len(A_candidates))
    b_idx = math.floor(random.random() * len(B_candidates))
    
#     print("A ind: {}".format(a_idx))
#     print("B ind: {}".format(b_idx))
    
    a_candidate = A_candidates[a_idx]
    b_candidate = B_candidates[b_idx]
    lines[3] = re.sub(lines[3].split()[-1], a_candidate, lines[3])
    lines[2] = re.sub(lines[2].split()[-1], b_candidate, lines[2])
    
    return lines
  elif scheme == "AABB":
    pass
  elif scheme == "ABBA":
    pass
  else:
    # invalid scheme for quatrain
    return None

def clean_poem(poem):
  poem = re.sub("eos", "", poem)
  poem = re.sub("<", "", poem)
  poem = re.sub("\n", "", poem)
  poem = re.sub(">", "\n", poem)

  lines = poem.split("\n")

  cleaned = []

  for l in lines:
    l = l.strip() # trailing whitespace
    cleaned.append(l)
    
  return cleaned

#poem = generate_quatrain("She")
#cleaned_poem = clean_poem(poem)
#cleaned_poem = list(filter(lambda l: len(l) > 1, cleaned_poem))
#
#print(cleaned_poem)
