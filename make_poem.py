import lang_model
from rhyme_model import *
from rhyme_index import *
import torch
import util



rm = RhymeModel(10)

if torch.cuda.is_available():
  rm.load_state_dict(torch.load("data/models/rhyme-v3.pth"))
  rm.cuda()
else:
  print("GPU not available")
  device = torch.device('cpu')
  rm.load_state_dict(torch.load("data/models/rhyme-v3.pth", map_location=device))

rm.eval()

rhyme_dataset = util.load_data()

lm = lang_model.build_model()

rhyme_words, rhyme_vectors = get_rhymes(rm, rhyme_dataset)
  
rhyme_idx = create_rhyme_index(rhyme_vectors)

# ------------------------------

def gen(seed):
  s = lang_model.generate_quatrain(seed, lm)
  
  poem = lang_model.clean_poem(s)
      
  rhymes = lang_model.enforce_rhyme(poem, "ABBA", rhyme_idx, rhyme_words, rhyme_vectors) 

  while rhymes == None:
    s = lang_model.generate_quatrain("The", lm)
    poem = lang_model.clean_poem(s)
    rhymes = lang_model.enforce_rhyme(poem, "ABBA", rhyme_idx, rhyme_words, rhyme_vectors)  

  return rhymes

#quatrain = gen("He", rm, lm, rhyme_dataset, rhyme_words, rhyme_vectors, rhyme_idx)

#print(quatrain)
