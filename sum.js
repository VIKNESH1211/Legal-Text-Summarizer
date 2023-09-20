document.getElementById("summarizeButton").addEventListener("click", function () {
    const inputText = document.getElementById("inputText").value;

    fetch("YOUR_API_ENDPOINT", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            text: inputText,
        }),
    })
    .then((response) => response.json())
    .then((data) => {
        const summaryResult = document.getElementById("summaryResult");
        summaryResult.innerText = data.summary;
    })
    .catch((error) => {
        console.error("Error:", error);
    });
});