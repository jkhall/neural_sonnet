import nmslib
import numpy as np
import util

def get_rhymes(model, rhyme_dataset):
  vectors = []
  flat_rhymes = [i for sublist in rhyme_dataset for i in sublist]
  rhyme_words = list(set(flat_rhymes))
  
  for i in range(len(rhyme_words)):
    vec = model.get_vector_rep(util.word2indices(rhyme_words[i], model.char2idx).unsqueeze(0))
    vec_cpu = vec.detach().cpu().numpy()
  
    vectors.append(vec_cpu.squeeze())
      
  rhyme_words = np.array(rhyme_words)
  rhyme_vectors = np.array(vectors)

  return rhyme_words, rhyme_vectors 

def create_rhyme_index(rhyme_vectors):
  index = nmslib.init(method='hnsw', space='cosinesimil')
  index.addDataPointBatch(rhyme_vectors)
  index.createIndex({'post': 2}, print_progress=True)

  return index
