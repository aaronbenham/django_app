function updatechart(score, list) {
    newscore = score
    console.log(score)
    console.log(list)
};

function updatechart() {
    StandardDeviationGraph.config.data.datasets[0].data.push(newscore);
    StandardDeviationGraph.config.data.datasets[0].data.shift();
    StandardDeviationGraph.update();
};

//$.ajax({
//  type: "POST",
//  url: "C:/Users/BenhamAaron/Documents/adversarial_AI/djangoapp/djangoapp/views.py",
//  data: {
//    param: "hello world",
//  }
//})
////.done((o) {
////   console.log(o)
////});