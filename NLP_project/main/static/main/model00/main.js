let availableTags = [`house`, `boat`, `the`]

function writePoem()
{
    // Make the prediction
    availableTags = [`house`, `boat`, `the`]

    for (let i = 0; i < 20; i++)
        availableTags.push(`New suggestion ${Math.random(100)}`)

    $("#main_text").autocomplete({
        source: availableTags
    })

}
