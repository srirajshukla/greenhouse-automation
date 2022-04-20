let model;

const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');


const loadModel = async function () {
    // const model_path = "file://D:/workspace/nodered/testingTF/fashionp/model.json";
    console.log(__dirname);
    // const model_path = "file://fashion-predictor/tomato/model.json";
    const model_path = "file://fashion-predictor/od/model.json";
    console.log("object model loaded")
    // model = await tf.loadLayersModel(model_path);
    model = await tf.loadGraphModel(model_path);
    // console.log(model);
    return model;
}


const processInput = function(msg) {
    console.log("got input data to be processed")
    // console.log(msg);
    
    const uint8array = new Uint8Array(msg);
    // console.log(uint8array);

    let imageTensor = tf.node.decodeImage(uint8array, 3);
    imageTensor = tf.image.resizeBilinear(imageTensor, [224, 224]);
    // imageTensor = tf.image.resizeBilinear(imageTensor, [299, 299]);

    // console.log(imageTensor);
    imageTensor = imageTensor.reshape([1,224,224, 3]);
    // imageTensor = imageTensor.reshape([1,299,299, 3]);

    // console.log(imageTensor);

    return imageTensor;
}

function preprocess(image) {
    const uint8array = new Uint8Array(image);
    const imageTensor = tf.node.decodeImage(uint8array, 3);
    const widthToHeight = imageTensor.shape[1] / imageTensor.shape[0];
    let squareCrop;
    if (widthToHeight > 1) {
      const heightToWidth = imageTensor.shape[0] / imageTensor.shape[1];
      const cropTop = (1-heightToWidth) / 2;
      const cropBottom = 1 - cropTop;
      squareCrop = [[cropTop, 0, cropBottom, 1]];
    } else {
      const cropLeft = (1-widthToHeight) / 2;
      const cropRight = 1 - cropLeft;
      squareCrop = [[0, cropLeft, 1, cropRight]];
    }
    // Expand image input dimensions to add a batch dimension of size 1.
    const crop = tf.image.cropAndResize(
        tf.expandDims(imageTensor), squareCrop, [0], [224, 224]);
    return crop.div(255);
  }


const processOutput = async function (prediction) {
    let output = prediction.dataSync();
    let maxindex = 0;
    for (let i = 1; i < output.length; i++) {
        if (output[i] > output[maxindex]) {
            maxindex = i;
        }
    }
    const classIndex = await tf.argMax(tf.squeeze(prediction)).data();
    const className = model.metadata['classNames'][classIndex[0]];

    return {
        label: "Prediction of the Image",
        predictionTensor: prediction,
        predictionVector: prediction.dataSync(),
        classIndex: classIndex,
        className: className,
    };
}


const xormodel = async function(a, b) {

    console.log(__dirname);
    const model_path = "file:///D:/workspace/nodered/testingTF/fashionp/model.json";
    try{
    const model = await tf.loadLayersModel(model_path);
    } catch (error){
        console.log(error);
    }
    console.log(model);

    // const imageBuffer = fs.readFileSync('D:/workspace/nodered/nodered-dev/fashion-predictor/fashionp/name.png')
    // const uint8array = new Uint8Array(imageBuffer);
    // let imageTensor = tf.node.decodeImage(uint8array, 1);
    // imageTensor = imageTensor.reshape([1, 28, 28])


    // console.log(imageTensor);
    // "D:\workspace\nodered\nodered-dev\fashion-predictor\fashionp\name.png"


    // const output = model.predict(imageTensor);

    // console.log(output)

    
    // const outputData = output.dataSync();
    // console.log(outputData)
    
    // const input2d = tf.tensor2d([[a,b]]);
    // const output = model.predict(input2d)
    // const outputData = output.dataSync();

    // console.log(`Probability = ${outputData}\tPredicted Value = ${Number(outputData[0] > 0.5)}`)
}
console.log(__dirname);
module.exports = {
    loadModel: loadModel,
    processInput: processInput,
    processOutput: processOutput,
    xormodel: xormodel,
    preprocess: preprocess
}