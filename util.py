import codecs
import itertools

# rhymes from line endings
print("hey")

def get_last_words():
  print("getting last words...")
  all_words = []
  for doc in codecs.open("data/train/sonnet_train.txt", "r", "utf-8"):

    word_lines, char_lines = [[], []], []
    last_words = []

    # in deepspeare, they reverse this
    for line in doc.strip().split("<eos>"):
      words = line.strip().split()

      if len(words) > 1:
        lw = words[-1]
        lw = re.sub("hyppphen", "-", lw)
        last_words.append(lw)
    
    all_words.append(last_words)
    
  return all_words
  
def load_data():
  
  data = [] # contains lists of [target, r1, r2, r3]
  lw = get_last_words()
  
  for endings in lw:
    # get quatrains
    
    n = 4 # four lines in a quatrain
    q1 = np.array(endings[:n])
    q2 = np.array(endings[n:2*n])
    q3 = np.array(endings[2*n:3*n])
    
    data.append(q1)
    data.append(q2)
    data.append(q3)
      
  return data

def word2indices(word):
  widxs = torch.zeros(max_word_len, dtype=torch.long).cuda()
  if len(word) > max_word_len: print(word) 
  for i, ch in enumerate(word):
    widxs[i] = char2idx[ch]
    
  return widxs

def prep_data(d):
  prepped = []
  for item in d:
    prepped.append((word2indices(item[0]), list(map(lambda w: word2indices(w), item[1:]))))    
  return prepped

