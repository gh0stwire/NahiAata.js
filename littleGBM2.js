//-------------------------------------------------------------------------------------
//                               Main Framework
//-------------------------------------------------------------------------------------
// Global variables.
let initGainSum, trainMax, trainMin, mem_itn;
let valid=true;
let gtree = []
//-------------------------------------------------------------------------------------
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
        .filter(row => row.trim() !== "") // Ignore empty lines
        .map(row => row.split(",").map(value => isNaN(value) ? value : Number(value)));
        
        if(data[0].length < data[1].length && valid) {
            valid=false;
            alert("Error: Insufficient headers");
        }
        else if (data[0].length>data[1].length && valid) {
            valid=false;
            alert("Error: Excess Headers");
        };  
    try{let keys = data[0].map(key => key.trim());} catch(error){alert("Error: Headers missing / purely numerical")}
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

//-------------------------------------------------------------------------------------
// Function to determine best splitting parameter
const fgain = (mdata) => {
    let splits = [];
    for (let j = 0; j < Object.keys(mdata[0]).length - 1; j++) {
        let feature = Object.keys(mdata[0])[j];
        msort(mdata, feature);
		fsplits=[-Infinity,'x',-Infinity,-Infinity];
        let out = mdata.map(row => row[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]);
        for (let i = 0; i < mdata.length - 1; i++) {
            let tsi = i;
            let tsp = mdata[i][feature];
            let gain = Math.pow((frss(out) - (frss(out.slice(0, i + 1)) * (i + 1) / out.length + frss(out.slice(i + 1)) * (out.length - (i + 1)) / out.length)), 0.5);
            if (fsplits[3] < gain) fsplits = [tsi, feature, tsp, gain];
        }
	  	splits.push(fsplits);
    }
    return splits;
};
//-------------------------------------------------------------------------------------
// Function to initialize first-step predictions 
const first_step = (mdata) => {
    let out = mdata.map(row => row[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]);
    let pred1 = avg(out);
    return mdata.map(row => ({ ...row, [Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]: pred1 }));
};
//-------------------------------------------------------------------------------------
// Recursive function to train the model
const track = (pred, mdata, lrnf= 0.1, itn=100, tree = []) => {
	let res = pred.map((row, i) => ({
        ...row,
        [Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]: mdata[i][Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]] - row[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]
    }));
	let splits = fgain(res);
 	let gain_sum = splits.reduce((acc, curr)=> acc + curr[3],0);
  	console.log(gain_sum);
  	if (mem_itn === itn) initGainSum = gain_sum;
  	if (gain_sum <= 0.2*initGainSum && valid){
		alert(`Training didn't require ${mem_itn} iterations. Ran ${mem_itn - itn} times.`);
	  	let out = pred.map(row => row[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]);
        trainMax = maxVal(out);
        trainMin = minVal(out);
	  	return {pred, tree};
	}
  	if (itn <= 0) {
	  let out = pred.map(row => row[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]);
      trainMax = maxVal(out);
	  trainMin = minVal(out);
	  return { pred, tree };
	};
    
  	let node = [];
	for (let i=0; i<splits.length; i++){
	  msort(res, splits[i][1]);
	  msort(pred, splits[i][1]);
	  msort(mdata, splits[i][1]);
	  let lres_sum = res.slice(0, splits[i][0] + 1).reduce((sum, row) => sum + row[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]], 0);
	  let rres_sum = res.slice(splits[i][0] + 1).reduce((sum, row) => sum + row[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]], 0);

	  let lres_avg = lres_sum / (splits[i][0] + 1);
	  let rres_avg = rres_sum / (res.length - splits[i][0] - 1);
	  let flrnf = lrnf*(1/(1))*(splits[i][3]/gain_sum);
	  let flcorr = flrnf*lres_avg;
	  let frcorr = flrnf*rres_avg;
	  
	  for (let j = 0; j < pred.length; j++) {
		  pred[j][Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]] += (j <= splits[i][0] ? flcorr : frcorr);
	  }
	  node.push([splits[i][1], splits[i][2], flcorr, frcorr]);
	}
	tree.push(node);
    return track(pred, mdata, lrnf, itn - 1, tree);
};

//-------------------------------------------------------------------------------------
// Function to make predictions
const predict = (input, mdata, tree) => {
    let data = modifier(input);
  
    // Store the original index to maintain order
    let indexedData = data.map((row, index) => ({ ...row, originalIndex: index }));
    
    let predictions = indexedData.map((row) => {
        let pred = avg(mdata.map(r => r[Object.keys(mdata[0])[Object.keys(mdata[0]).length - 1]]));

        for (let j = 0; j < tree.length; j++) {
            for (let i = 0; i < tree[j].length; i++) {
                row[tree[j][i][0]] <= tree[j][i][1] ? pred += tree[j][i][2] : pred += tree[j][i][3];
            }
        }

        let normalizedPred = ((pred - trainMin) / (trainMax - trainMin)).toFixed(3);
        if (normalizedPred <=0) normalizedPred = 0;
        if (normalizedPred >=1) normalizedPred = 1;
        return { ...row, Prediction: normalizedPred };
    });

    // Restore the original order
    predictions.sort((a, b) => a.originalIndex - b.originalIndex);
    return predictions.map(({ originalIndex, ...rest }) => rest);
};

//-------------------------------------------------------------------------------------
//                                 Integrating with HTML
//-------------------------------------------------------------------------------------
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
    valid=true;
    const csvText = document.getElementById("text-input");
    const csvFile = document.getElementById("input-handler");
    lrnf=document.querySelector('#lrnf').value || 0.1;
    itn=document.querySelector('#depth').value || 100;
    mem_itn = itn;
    if (csvFile.disabled){
        let {pred, tree} = track(first_step(modifier(csvText.value)), modifier(csvText.value), lrnf, itn)
        gtree=tree;
        if (tree.length > 0 && valid) alert("Decision tree formed successfully!");
    } else {
        const file = csvFile.files[0];
        if (!file) {
            alert("Please select a CSV file.");
            return;
        }

        const reader = new FileReader();
        reader.onload = function (e) {
            const csvData = e.target.result;
            let {pred, tree} = track(first_step(modifier(csvData)), modifier(csvData), lrnf, itn);
            gtree=tree;
        if (tree.length>0 && valid) alert("Decision tree formed successfully!");
        }

        reader.readAsText(file)
    }
})

document.querySelector('#predict-submit-button').addEventListener('click', () => {
    valid=true;
    const csvText = document.getElementById("text-input");
    const csvFile = document.getElementById("input-handler");
    let data;

    if (csvFile.disabled) {
        data = predict(document.querySelector('#prediction-input').value, modifier(csvText.value), gtree);
        if(valid){
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
            if (valid){
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
    gtree=[];
    initGainSum=undefined;
    trainMax=-Infinity;
    trainMin=+Infinity;
    valid = true;
});

document.getElementById("reset-output-button").addEventListener("click", () => {
    document.getElementById("prediction-input").value = ""; 
    document.getElementById("prediction-output").innerHTML = ""; 
    valid=true;
});
