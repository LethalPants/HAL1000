let errorContainer = document.getElementById('error-container');
let output = document.getElementById('output');
document.getElementById('image').addEventListener('change', (ev) => {
    output.innerHTML = `<h3>Lemme think...</h3>`;
    let imageBox = document.getElementById('image-box');
    let image = ev.target.files[0];
    imageBox.innerHTML = `<img src=${URL.createObjectURL(
        image
    )} alt="picture">`;
    const formData = new FormData();
    formData.append('image', image);

    fetch('/predict', {
        method: 'POST',
        body: formData,
    })
        .then((res) => res.json())
        .then((data) => makeSentence(data))
        .catch((err) => {
            errorContainer.innerHTML = `<div class="error"> Something went wrong. Please reload the page and try again. </div>`;
            console.log(err);
        });
});

function makeSentence(data) {
    if (data.success === true) {
        let predLabel = data.predictions[0].label;
        let prob = Math.round(data.predictions[0].probability * 100);
        predLabel = predLabel.replace(/_/g, ' ');
        output.innerHTML = `<h3> I'm ${prob}% sure that is a ${predLabel}</h3>`;
    } else {
        errorContainer.innerHTML = `<div class="error"> Something went wrong. Please reload the page and try again. </div>`;
    }
}
