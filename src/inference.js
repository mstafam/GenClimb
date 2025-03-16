/*global BigInt */
/*global BigInt64Array */

const ort = require('onnxruntime-web');
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";

const token_to_id_path = './static/token_to_id.json';
const id_to_token_path = './static/id_to_token.json';
const placements_path = './static/placements.json';
const roles_path = './static/roles.json';
const layoutInfo_path = './static/layoutInfo.json';
const model_path = './static/GenClimb_Tiny_int8.onnx';

const vocab_size = 1279;
let regen = 0;
let downLoadingModel = true;

// WASM configuration
ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = false;

// Inference Session Options
const options = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
};

// Initialize model session
let session;
try {
  console.log("Attempting to load model from:", model_path);
  session = ort.InferenceSession.create(model_path, options);
  session.then(t => { 
    downLoadingModel = false;
    console.log("Model loaded successfully");
    // warmup inference
    console.log("Warming up the model...");
  }).catch(err => {
    console.error("Error loading model:", err);
    downLoadingModel = false;
  });
} catch (err) {
  console.error("Error initializing session:", err);
  downLoadingModel = false;
}

// Convert Tokens to Tensors
async function getIds(src, tgt) {
    const src_dims = [1, src.length];
    const tgt_dims = [1, tgt.length];

    const token_to_id = await fetch(token_to_id_path).then(d => d.json());
    let srcTensors = [];
    let tgtTensors = [];

    // Convert each token to id
    for (let i = 0; i < Math.max(src_dims[1], tgt_dims[1]); i++) {
        if (i < src_dims[1]) {
            srcTensors.push(BigInt(token_to_id[src[i]]));
        }
        if (i < tgt_dims[1]) {
            tgtTensors.push(BigInt(token_to_id[tgt[i]]));
        }
    }

    // Create Tensors
    var src_ids = new ort.Tensor('int64', BigInt64Array.from(srcTensors), src_dims);
    var tgt_ids = new ort.Tensor('int64', BigInt64Array.from(tgtTensors), tgt_dims);

    // Return feed
    return {
        src: src_ids,
        tgt: tgt_ids
    }
}

// Convert Ids to Tokens
async function getTokens(tgt_ids) {
    const id_to_token = await fetch(id_to_token_path).then(d => d.json());
    let tokens = [];

    // Convert each Id to token
    for (let i = 0; i < tgt_ids.length; i++) {
        tokens.push(id_to_token[Number(tgt_ids[i])]);
    }

    return tokens
}

// Ensure valid tokens and [placement, role] pattern
function initalOutputValidation(frame) {
    // Ensure there is no <PAD>, < SOS >, <EOS> tokens
    if (frame.includes('<PAD>') || frame.includes('< SOS >') || frame.includes('<EOS>')) {
        return false
    }

    // Ensure it follow placement, role pattern e.g [P1111, R22, ...]
    for (let i = 0; i < frame.length; i+=2) {
        if (i + 1 >= frame.length || frame[i][0] !== 'p' || frame[i+1][0] !== 'r') {
            return false
        }
    }

    return true
}

// Strict Validation
function strictOutputValidation(circles, layoutInfo) {
    if (!circles || !layoutInfo) {
        return false;
    }
    
    const edge_left = layoutInfo['edge_left'];
    const edge_right = layoutInfo['edge_right'];
    const edge_bottom = layoutInfo['edge_bottom'];
    const edge_top = layoutInfo['edge_top'];

    // Ensure layoutInfo has all required properties
    if (edge_left === undefined || edge_right === undefined || 
        edge_bottom === undefined || edge_top === undefined) {
        return false;
    }

    // Holds must be within the edges.
    let x_vals = [];
    let y_vals = [];
    let colors = [];

    for (let i = 0; i < circles.length; i++) {
        x_vals.push(parseInt(circles[i][0]));
        y_vals.push(parseInt(circles[i][1]));
        colors.push(circles[i][2]);
    }

    const max_x = Math.max(...x_vals);
    const min_x = Math.min(...x_vals);
    const max_y = Math.max(...y_vals);
    const min_y = Math.min(...y_vals);

    // Ensure all holds are within the edges of the board
    if (
        min_x < edge_left || max_x > edge_right || min_y < edge_bottom || max_y > edge_top
    ) {
        return false
    }

    // There must atleast 1 start hold, max of 2.
    const num_start = colors.filter(el => el === '#00FF00').length;
    if (1 > num_start || num_start > 2) {
        return false
    }

    // There must atleast 1 finish hold, max of 2.
    const num_finish = colors.filter(el => el === '#FF00FF').length;
    if (1 > num_finish || num_finish > 2) {
        return false
    }

    return true
}

// Top-P Sampling
function top_p(logits, p) {
    const probs = softmax(logits);
    const indexed = probs.map((prob, index) => [prob, index]);
    indexed.sort((a, b) => b[0] - a[0]);
    let cumSum = 0;
    const newLogits = new Array(logits.length).fill(0);
    
    for (const [prob, index] of indexed) {
        if (cumSum >= p) break;
        newLogits[index] = logits[index]; // Use original logit value
        cumSum += prob;
    }
    
    return newLogits
}

// Top-K Sampling
function top_k(logits, k) {
    const indexed = logits.map((value, index) => [value, index]);
    indexed.sort((a, b) => b[0] - a[0]);
    const topK = indexed.slice(0, k);
    const newLogits = new Array(logits.length).fill(0);

    topK.forEach(([value, index]) => {
        newLogits[index] = value;
    });
    
    return newLogits
}

// Multinomial Sampling
function multinomial(input, num_samples) {
    function sampleRow(probs, num_samples) {
        const indices = [];
        for (let i = 0; i < num_samples; i++) {
            const rand = Math.random();
            let cumSum = 0;
            for (let j = 0; j < probs.length; j++) {
                cumSum += probs[j];
                if (rand < cumSum) {
                    indices.push(j);
                    break;
                }
            }
        }
        return indices
    }

    return sampleRow(input, num_samples)
}

// Softmax function
function softmax(logits) {
    let normalized_logits = []
    let denom = 0;
    for (let i = 0; i < logits.length; i++) {
        const logit_exp = Math.exp(logits[i]);
        denom += logit_exp;
        normalized_logits.push(logit_exp);
    }
    return normalized_logits.map(val => val / denom);
}

// Determine holds in the climb
async function getCircles(frames) {
    // [[x,y, color], .....]
    try {
        const placements = await fetch(placements_path).then(d => d.json());
        const roles = await fetch(roles_path).then(d => d.json());
        let circles = [];
        
        for (let i=0; i < frames.length; i+=2) {
            if (i + 1 >= frames.length) {
                console.error("Incomplete frame data");
                return null;
            }
            
            let placement_id = String(frames[i].substring(1));
            let res1 = placements[placement_id];
            
            let role_id = String(frames[i+1].substring(1));
            let res2 = roles[role_id];
            
            if (!res1 || !res2) {
                console.error("Missing placement or role data", placement_id, role_id);
                return null;
            }
            
            let color, x, y;
            try {
                color = "#" + res2['color'];
                x = String(parseInt(res1['x']));
                y = String(parseInt(res1['y']));
            } catch (err) {
                console.error("Error parsing circle data:", err);
                return null;
            }
            
            let circle_data = [x, y, color];
            circles.push(circle_data);
        }
        
        return circles;
    } catch (err) {
        console.error("Error fetching placement or role data:", err);
        return null;
    }
}

// Get layout board specs
async function getLayoutInfo(layout) {
    // [layoutName, width, height, edge_left, edge_right, edge_bottom, edge_top]
    try {
        const layoutInfo = await fetch(layoutInfo_path).then(d => d.json());
        let layoutName = layout.replace(/\s+/g, '');
        return layoutInfo[layoutName] || null;
    } catch (err) {
        console.error("Error fetching layout info:", err);
        return null;
    }
}

// Check if model is downloading
function isDownloading() {
    return downLoadingModel;
}

// Inference function
export async function Inference(src, tgt, temp, samplingMethod, k=5, p=0.9, modelName, validationType) {
    if (isDownloading()) {
        return ["Model still loading, please wait..."];
    }
    
    // Target Sequence Length
    const seq_len = tgt.length;

    try {
        // Get Ids from the tokens
        const model_input = await getIds(src, tgt);

        if (!session) {
            console.error("Model session not initialized properly");
            return ["Model session not initialized properly. Please check console for errors."];
        }

        const sessionInstance = await session.catch(err => {
            console.error("Error accessing session:", err);
            throw new Error("Failed to access model session: " + err.message);
        });
        
        if (!sessionInstance) {
            console.error("Failed to load model session");
            return ["Failed to load model session. Please check console for errors."];
        }
        
        // Run the session
        const output = await sessionInstance.run(model_input);
        
        // Get Last Token logits and scale with temperature
        const logits = Array.from(output['output'].data);
        const lastTokenLogits = logits.slice((seq_len - 1) * vocab_size, seq_len * vocab_size);
        const scaled_logits = lastTokenLogits.map(logit => logit / temp);

        let sampledLogits;

        // Sample logits with samplingMethod
        if (samplingMethod === 'top_k') {
            sampledLogits = top_k(scaled_logits, k);
        } else if (samplingMethod === 'top_p') {
            sampledLogits = top_p(scaled_logits, p);
        } else {
            sampledLogits = scaled_logits;
        }

        // Get the next token ID
        const probabilities = softmax(sampledLogits);
        let next_token_id = multinomial(probabilities, 1);
        
        // Convert token ID to token
        let next_token = await getTokens(next_token_id);

        // Add token to existing Target Sequence
        let new_tgt = tgt.concat(next_token);

        // Autoregressively generate tokens until EOS token
        if (next_token[0] !== "<EOS>") {
            return await Inference(src, new_tgt, temp, samplingMethod, k, p, modelName, validationType)
        }
        
        // Final target sequence, with SOS and EOS tokens removed
        const final_tgt = new_tgt.slice(1, new_tgt.length-1)

        // Basic Validation
        const initalValidation = initalOutputValidation(final_tgt);

        // If basic validation is passed
        if (initalValidation) {
            // Get circles and layout information
            const circles = await getCircles(final_tgt);
            if (!circles) {
                // Handle error in getCircles
                console.error("Error in getCircles for final_tgt:", final_tgt);
                regen += 1;
                return await Inference(src, ['< SOS >'], temp, samplingMethod, k, p, modelName, validationType);
            }
            
            const layoutInfo = await getLayoutInfo(src[0]);
            if (!layoutInfo) {
                console.error("Error in getLayoutInfo for src[0]:", src[0]);
                regen += 1;
                return await Inference(src, ['< SOS >'], temp, samplingMethod, k, p, modelName, validationType);
            }
            
            // Strict Validation
            if (validationType === "strict") {
                const strictValidation = strictOutputValidation(circles, layoutInfo);
                // If strict Validation is not passed, regenerate climb from start
                if (!strictValidation) {
                    console.log("Failed strict validation, regenerating...");
                    regen += 1;
                    return await Inference(src, ['< SOS >'], temp, samplingMethod, k, p, modelName, validationType)
                }
            }
            
            // Return climb information if valid
            return [final_tgt, circles, layoutInfo, regen]
        }
        
        // Regenerate climb from start if generated one is invalid
        console.log("Failed initial validation, regenerating...");
        regen += 1;
        return await Inference(src, ['< SOS >'], temp, samplingMethod, k, p, modelName, validationType)
    } catch (error) {
        console.error("Error during inference:", error);
        return ["Error during inference: " + error.message];
    }
}

// Check model and WASM availability
export function checkEnvironment() {
    // Check if WebAssembly is supported
    if (typeof WebAssembly === 'undefined') {
        return {
            supported: false,
            error: "WebAssembly is not supported in this browser"
        };
    }
    
    return {
        supported: true,
        modelLoading: isDownloading(),
        modelPath: model_path
    };
}