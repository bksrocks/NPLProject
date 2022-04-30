let availableTags = []

const modelLoc = "./model00/"

async function createModel(){
    const model = await tf.loadLayersModel("https://poem-maker.s3.amazonaws.com/static/main/model00/model.json");
    console.log(model.summary())
    return model
}

function getTokenisedWord(seedText) {
    const token = modelLoc + "token.json";
    const wordToken = token[seedText.toLowerCase()];
    return tf.tensor1d([wordToken])
}

async function returnBestWords(seedText){
    const bestWords = await createModel()

    for (let i = 0; i < 20; i++){
        const seedWordToken = getTokenisedWord(seedText);
        bestWords.predict(seedWordToken, verbose=0).data().then(predictions => {
        const resultIdx = tf.argsort(predictions).dataSync()[0];
        availableTags.appendData(token[resultIdx]);
        })
    }
}

function writePoem()
{
    let seedText = $("#main_text");
    availableTags = ['the', 'hello', 'goodmorning'];
    bestWords = returnBestWords(seedText);
    for (let i = 0; i < 20; i++)
        availableTags.push()

    $("#main_text").autocomplete({
        source: availableTags
    })
}
