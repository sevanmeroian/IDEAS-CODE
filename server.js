const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const natural = require('natural');
const fs = require('fs');

// Initialize Express app
const app = express();
app.use(bodyParser.json());
app.use(cors());

// Placeholder for the trained model
let model;
let vocab = {};
let maxSeqLength = 0;

// Load or Train the Model
async function trainOrLoadModel() {
    if (fs.existsSync('./models/news_model.json')) {
        console.log('Loading saved model...');
        model = await tf.loadLayersModel('file://./models/news_model.json');
        vocab = JSON.parse(fs.readFileSync('./vocab.json', 'utf8'));
        maxSeqLength = vocab.maxSeqLength;
        console.log('Model loaded successfully.');
    } else {
        console.log('Training a new model...');
        await trainModel();
    }
}

// Train Model
async function trainModel() {
    // Load dataset
    const trueData = loadCSV('./True.csv', 1); // 1 for True News
    const fakeData = loadCSV('./Fake.csv', 0); // 0 for Fake News
    const combinedData = [...trueData, ...fakeData].sort(() => Math.random() - 0.5);

    // Tokenize and preprocess text
    const tokenizer = new natural.WordTokenizer();
    const vocabSet = new Set();
    combinedData.forEach(item => {
        item.tokens = tokenizer.tokenize(item.text.toLowerCase());
        item.tokens.forEach(word => vocabSet.add(word));
    });

    // Build vocabulary
    vocab = Array.from(vocabSet).reduce((obj, word, index) => {
        obj[word] = index + 1;
        return obj;
    }, {});
    vocab.maxSeqLength = Math.max(...combinedData.map(item => item.tokens.length));
    maxSeqLength = vocab.maxSeqLength;

    // Save vocabulary
    fs.writeFileSync('./vocab.json', JSON.stringify(vocab));

    // Convert text to sequences
    combinedData.forEach(item => {
        item.sequence = item.tokens.map(token => vocab[token] || 0);
        while (item.sequence.length < maxSeqLength) item.sequence.push(0); // Pad sequences
    });

    // Split data
    const trainSize = Math.floor(combinedData.length * 0.8);
    const trainData = combinedData.slice(0, trainSize);
    const testData = combinedData.slice(trainSize);

    const xTrain = tf.tensor2d(trainData.map(item => item.sequence));
    const yTrain = tf.tensor2d(trainData.map(item => [item.label]));
    const xTest = tf.tensor2d(testData.map(item => item.sequence));
    const yTest = tf.tensor2d(testData.map(item => [item.label]));

    // Define the model
    model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: Object.keys(vocab).length + 1, outputDim: 50 }));
    model.add(tf.layers.lstm({ units: 64, returnSequences: false }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });

    // Train the model
    await model.fit(xTrain, yTrain, {
        epochs: 10,
        validationData: [xTest, yTest],
        batchSize: 32,
    });

    // Save the model
    await model.save('file://./models/news_model');
    console.log('Model trained and saved successfully.');
}

// Load CSV Data
function loadCSV(filePath, label) {
    const csv = fs.readFileSync(filePath, 'utf8');
    const lines = csv.split('\n').slice(1); // Skip header
    return lines.map(line => ({ text: line, label }));
}

// Predict Credibility
async function predictNews(text) {
    const tokenizer = new natural.WordTokenizer();
    const tokens = tokenizer.tokenize(text.toLowerCase());
    const sequence = tokens.map(token => vocab[token] || 0);

    while (sequence.length < maxSeqLength) sequence.push(0); // Pad sequence

    const inputTensor = tf.tensor2d([sequence]);
    const prediction = model.predict(inputTensor).dataSync()[0];
    return prediction * 100; // Return as percentage
}

// API Endpoint: Analyze News
app.post('/analyze', async (req, res) => {
    try {
        const { text } = req.body;

        if (!text || text.trim() === '') {
            return res.status(400).json({ success: false, error: 'No text provided' });
        }

        const credibilityScore = await predictNews(text);
        res.json({
            success: true,
            credibility_score: credibilityScore.toFixed(2),
        });
    } catch (error) {
        console.error('Error analyzing text:', error);
        res.status(500).json({ success: false, error: 'Internal server error' });
    }
});

// Start the Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, async () => {
    console.log(`Server running on port ${PORT}`);
    await trainOrLoadModel();
});
