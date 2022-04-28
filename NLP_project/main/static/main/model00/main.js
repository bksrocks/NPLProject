import * as tf from '@tensorflow/tfjs';
import * as model from './model';


let availableTags = []

const model = await tf.loadLayersModel('NLP_project/main/static/main/model00/model.json');

function writePoem()
{

    let seed_text = $("#main_text")
    availableTags = []
    best_words = return_best_words(seed_text, {
        "num_of_words": 3
    });
    for (let i = 0; i < 20; i++)
        availableTags.push()

    $("#main_text").autocomplete({
        source: availableTags
    })

}
