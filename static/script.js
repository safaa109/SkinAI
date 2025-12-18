document.getElementById("uploadForm").addEventListener("submit", function(e) {
    e.preventDefault(); // يمنع Reload

    const fileInput = document.getElementById("imageInput");
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    document.getElementById("loading").style.display = "block";
    document.getElementById("result").innerHTML = "";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("loading").style.display = "none";

        document.getElementById("result").innerHTML = `
            <img src="${data.image}" width="200"><br><br>
            <b>Disease:</b> ${data.disease}<br><br>
            <b>Cause:</b> ${data.cause}<br><br>
            <b>Prevention:</b> ${data.prevention}<br><br>
            <b>Treatment:</b> ${data.treatment}
        `;
    })
    .catch(err => {
        document.getElementById("loading").innerText = "Error!";
        console.error(err);
    });
});
