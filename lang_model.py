import fastai
from fastai.text import *

data_lm = TextLMDataBunch.from_folder("data")
learn = language_model_learner(data_lm, text.models.awd_lstm, drop_mult=0.5)
learn.load(r"awd-v1")

def generate_quatrain(seed, lm=learn):
  if lm != None:
    s = lm.predict(seed, n_words=40)

    while s.count("eos") < 4:
      next_word = re.sub(s, "", lm.predict(s, n_words=1))
      s += next_word
      
    return s

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
