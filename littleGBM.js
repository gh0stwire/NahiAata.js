// Global variables removed
let gmaxTP;
let gminTP;
let gtree = []
let lrnf=0.1;
let depth=100;
// Predefined utility functions
const range = (n) => Array.from({ length: n }, (_, i) => i);
const msort = (mdata, feature) => mdata.sort((a, b) => a[feature] - b[feature]);
const frss = (arr) => {
    let mean = arr.reduce((sum, num) => sum + num, 0) / arr.length;
    let squaredDifferences = arr.map(num => Math.pow(num - mean, 2));
    return squaredDifferences.reduce((sum, squaredDiff) => sum + squaredDiff, 0);
};
const avg = (arr) => arr.reduce((sum, num) => sum + num, 0) / arr.length;
const minVal = (arr) => arr.reduce((min, num) => Math.min(min, num), Infinity);
const maxVal = (arr) => arr.reduce((max, num) => Math.max(max, num), -Infinity);

// Function to convert CSV data into an array of objects
const modifier = (strg) => {
    const data = strg
        .split("\n")
        .map(row => row.split(",").map(value => isNaN(value) ? value.trim() : Number(value)))
        .filter(row => row.length > 1 && row.some(value => value !== "")); // Removes empty rows

    try {
        let keys = data[0].map(key => key.trim());
    } catch (error) {
        alert("Error: Please enter non-empty data. Also ensure headers are present.");
    }

    let keys = data[0].map(key => key.trim());
    let mdata = data.slice(1).map(row => Object.fromEntries(keys.map((key, i) => [key, row[i]])));

    return mdata.map(item => {
        let cleanedItem = {};
        for (let key in item) {
            cleanedItem[key.trim()] = item[key];
        }
        return cleanedItem;
    });
};


// Example dataset
let a = `Age,Gender,Hospitalizations,Family History,Substance Abuse,Social Support,Stress Factor,Medication Adherence,Diagnosis
72,1,0,0,0,0,2,2,0
49,1,1,1,1,2,2,0,1
53,1,0,1,0,0,1,1,1
67,1,0,0,1,1,1,2,0
54,0,0,0,0,0,1,0,0
65,0,0,0,0,2,2,0,0
31,0,0,0,0,0,0,1,0
44,0,1,1,0,1,0,2,1
76,0,0,0,1,0,2,2,0
36,1,10,0,1,2,0,1,1`;

// Function to determine best splitting parameter
const fgain = (mdata) => {
    let best_split_data = [0, "Hello World", 0, 0];
    for (let j = 0; j < Object.keys(mdata[0]).length - 1; j++) {
        let feature = Object.keys(mdata[0])[j];
        msort(mdata, feature);

        let out = mdata.map(row => row[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]);
        for (let i = 0; i < mdata.length - 1; i++) {
            let tsi = i;
            let tsp = mdata[i][feature];
            let gain = frss(out) - (frss(out.slice(0, i + 1)) * (i + 1) / out.length + frss(out.slice(i + 1)) * (out.length - (i + 1)) / out.length);
            if (best_split_data[3] < gain) best_split_data = [tsi, feature, tsp, gain];
        }
    }
    return best_split_data;
};

// Function to initialize first-step predictions
const first_step = (mdata) => {
    let out = mdata.map(row => row[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]);
    let pred1 = avg(out);
    return mdata.map(row => ({ ...row, [Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]: pred1 }));
};

// Recursive function to train the model
const track = (pred, mdata, lrnf, depth, tree = []) => {
    if (depth <= 0) return { pred, tree };

    let res = pred.map((row, i) => ({
        ...row,
        [Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]: mdata[i][Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]] - row[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]
    }));

    let best_split = fgain(res);
    msort(res, best_split[1]);
	msort(pred, best_split[1]);
  	msort(mdata, best_split[1]);
    let lres_sum = res.slice(0, best_split[0] + 1).reduce((sum, row) => sum + row[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]], 0);
    let rres_sum = res.slice(best_split[0] + 1).reduce((sum, row) => sum + row[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]], 0);

    let lres_avg = lres_sum / (best_split[0] + 1);
    let rres_avg = rres_sum / (res.length - best_split[0] - 1);

    

    for (let i = 0; i < pred.length; i++) {
        pred[i][Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]] += (i <= best_split[0] ? lrnf * lres_avg : lrnf * rres_avg);
    }

    tree.push([best_split[1], best_split[2], lrnf * lres_avg, lrnf * rres_avg]);

    return track(pred, mdata, lrnf, depth - 1, tree);
};


// Function to make predictions
const predict = (input, mdata, tree) => {
    let data = modifier(input);
    
    // Store the original index to maintain order
    let indexedData = data.map((row, index) => ({ ...row, originalIndex: index }));
    
    let predictions = indexedData.map((row) => {
        let pred = avg(mdata.map(r => r[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]));
        
        for (let j = 0; j < tree.length; j++) {
            row[tree[j][0]] <= tree[j][1] ? pred += tree[j][2] : pred += tree[j][3];
        }
        
        return { ...row, Prediction: pred.toFixed(2) };
    });

    // Restore the original order
    predictions.sort((a, b) => a.originalIndex - b.originalIndex);
    
    // Remove the temporary originalIndex key
    return predictions.map(({ originalIndex, ...rest }) => rest);
};


//Arnav believes in Karma.
//Aaditya does too.
//So does Swara.


function toggleInputs() {
    const csvText = document.getElementById("text-input");
    const csvFile = document.getElementById("input-handler");

    if (csvText.value.trim() !== "") {
        csvFile.disabled = true;
    } else {
        csvFile.disabled = false;
    }

    if (csvFile.files.length > 0) {
        csvText.disabled = true;
    } else {
        csvText.disabled = false;
    }
}

document.querySelector('#input-submit-button').addEventListener('click', ()=>{
    const csvText = document.getElementById("text-input");
    const csvFile = document.getElementById("input-handler");
    lrnf=document.querySelector('#lrnf').value;
    depth=document.querySelector('#depth').value;
    if (csvFile.disabled){
        //console.log(modifier(csvText.value));
        let {pred, tree} = track(first_step(modifier(csvText.value)), modifier(csvText.value), lrnf, depth)
        gtree=tree;
        console.log(pred);
        if (tree.length > 0) alert("Data submitted successfully!");
    } else {
        const file = csvFile.files[0];
        if (!file) {
            alert("Please select a CSV file.");
            return;
        }

        const reader = new FileReader();
        reader.onload = function (e) {
            const csvData = e.target.result;
            let {pred, tree} = track(first_step(modifier(csvData)), modifier(csvData), lrnf, depth);
            gtree=tree;
            console.log(pred);
        if (tree.length>0) alert("Data submitted successfully!");
        }

        reader.readAsText(file)
    }
})

document.querySelector('#predict-submit-button').addEventListener('click', () => {
    console.log(gtree);
    const csvText = document.getElementById("text-input");
    const csvFile = document.getElementById("input-handler");
    let data;

    if (csvFile.disabled) {
        data = predict(document.querySelector('#prediction-input').value, modifier(csvText.value), gtree);
        
        // Table rendering (unchanged)
        const outputDiv = document.getElementById("prediction-output");
        const headers = Object.keys(data[0]);
        let tableHTML = "<table border='1'><tr>";
        tableHTML += headers.map(header => `<th>${header}</th>`).join("") + "</tr>";
        tableHTML += data.map(row => 
            "<tr>" + headers.map(header => `<td>${row[header]}</td>`).join("") + "</tr>"
        ).join("");
        tableHTML += "</table>";
        outputDiv.innerHTML = tableHTML;
    }

    if (!csvFile.disabled) {
        // Read CSV file and store in 'data'
        const file = csvFile.files[0];
        if (!file) {
            alert("Please select a CSV file.");
            return;
        }

        const reader = new FileReader();
        reader.onload = function (e) {
            const csvData = e.target.result;
            data = predict(document.querySelector('#prediction-input').value, modifier(csvData), gtree);
            // Table rendering (unchanged)
            const outputDiv = document.getElementById("prediction-output");
            const headers = Object.keys(data[0]);
            let tableHTML = "<table border='1'><tr>";
            tableHTML += headers.map(header => `<th>${header}</th>`).join("") + "</tr>";
            tableHTML += data.map(row => 
                "<tr>" + headers.map(header => `<td>${row[header]}</td>`).join("") + "</tr>"
            ).join("");
            tableHTML += "</table>";
            outputDiv.innerHTML = tableHTML;
        };

        reader.readAsText(file);
    }
});

document.getElementById("reset-data-button").addEventListener("click", () => {
    const csvText = document.getElementById("text-input");
    const csvFile = document.getElementById("input-handler");
    document.getElementById("input-handler").value = ""; 
    document.getElementById("text-input").value = ""; 
    document.querySelector("#lrnf").value="";
    document.querySelector("#depth").value="";
    csvFile.disabled=false;
    csvText.disabled=false;
    lrnf=0.1;
    depth=100;
    gtree=[];
});

document.getElementById("reset-output-button").addEventListener("click", () => {
    document.getElementById("prediction-input").value = ""; 
    document.getElementById("prediction-output").innerHTML = ""; 
});
