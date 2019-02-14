btn = document.getElementById("gen_poem_btn")
inp = document.getElementById("seed_inp")
display = document.getElementById("display")

btn.addEventListener("click", function(e){
  e.preventDefault()
  console.log("generating")
  display.innerHTML = "generating..."
  if(inp.value.length > 10){
    display.innerHTML = "seed length should be <= 10"
  } else {
    fetch("/gen_poem?seed=" + inp.value)
    .then(function(response){
      return response.json()
    })
    .then(function(myJson){
      let result = ""
      
      display.innerHTML = myJson.join('<br>')
    })
  } 
})
