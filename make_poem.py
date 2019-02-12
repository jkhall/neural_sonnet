import lang_model
from rhyme_model import *

rm = RhymeModel(10)
rm.load_state_dict(torch.load("data/models/rhyme-v3.pth"))
rm.cuda()
rm.eval()


