from starlette.applications import Starlette
from starlette.responses import PlainTextResponse, HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from rhyme_model import *
from rhyme_index import *
from make_poem import gen
import util
import lang_model
import uvicorn
import sys


app = Starlette()
app.debug = True
app.mount('/static', StaticFiles(directory="static"))

# build rhyme and language models
#rm = RhymeModel(10)
#
#if torch.cuda.is_available():
#  rm.load_state_dict(torch.load("data/models/rhyme-v3.pth"))
#  rm.cuda()
#else:
#  print("GPU not available")
#  device = torch.device('cpu')
#  rm.load_state_dict(torch.load("data/models/rhyme-v3.pth", map_location=device))
#
#rm.eval()
#
#rhyme_dataset = util.load_data()
#
#lm = lang_model.build_model()
#
#rhyme_words, rhyme_vectors = get_rhymes(rm, rhyme_dataset)
#  
#rhyme_idx = create_rhyme_index(rhyme_vectors)
#
# ------------------------------


@app.route("/")
def homepage(request):
  #return PlainTextResponse('Hello, world')
  return HTMLResponse("""
    <head>
      <link rel='stylesheet' href='static/main.css'/>
    </head>
    <input type='text' id='seed_inp' placeholder='Seed poem here (10 chars or less)'></input>
    <button id='gen_poem_btn'>Generate!</button>
    <div id='display'></div>
    <script src='static/main.js'></script>
  """)

@app.route("/gen_poem")
def poemgen(request):
  seed = request.query_params["seed"] if "seed" in request.query_params else "The" 
  poem = gen(seed)
  return JSONResponse(poem)


if __name__ == "__main__":
  if "serve" in sys.argv:
    uvicorn.run(app, host="0.0.0.0", port=80)
