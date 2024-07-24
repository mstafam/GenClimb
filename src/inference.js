/*global BigInt */
/*global BigInt64Array */

import * as ort from 'onnxruntime-web/webgpu';
import * as math from 'mathjs';
import { exit } from 'process';

const token_to_id_path = './static/token_to_id.json';
const id_to_token_path = './static/id_to_token.json';
const placements_path = './static/placements.json';
const roles_path = './static/roles.json';
const layoutInfo_path = './static/layoutInfo.json';
const model_large_url= "./static/GenClimb_Large_int8.onnx";
const model_tiny_url = "./static/GenClimb_Tiny_int8.onnx";
const vocab_size = 1279;
let regen = 0;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

// Inference Session Options
const options = {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all'
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
    // Ensure there is no <PAD>, <SOS>, <EOS> tokens
    if (frame.includes('<PAD>') || frame.includes('<SOS>') || frame.includes('<EOS>')) {
        return false
    }

    // Ensure it follow placement, role pattern e.g [P1111, R22, ...]
    for (let i = 0; i < frame.length; i+=2) {
        if (frame[i][0] !== 'p' || frame[i+1][0] !== 'r') {
            return false
        }
    }

    return true
}

// Strict Validation
function strictOutputValidation(circles, layoutInfo) {
    const edge_left = layoutInfo['edge_left'];
    const edge_right = layoutInfo['edge_right'];
    const edge_bottom = layoutInfo['edge_bottom'];
    const edge_top = layoutInfo['edge_top'];

    // Holds must be within the edges.
    let x_vals = [];
    let y_vals = [];
    let colors = [];

    for (let i = 0; i < circles.length; i++) {
        x_vals.push(circles[i][0]);
        y_vals.push(circles[i][1]);
        colors.push(circles[i][2]);
    }

    const max_x = math.max(x_vals);
    const min_x = math.min(x_vals);
    const max_y = math.max(y_vals);
    const min_y = math.min(y_vals);

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
        const logit_exp = math.exp(logits[i]);
        denom += logit_exp;
        normalized_logits.push(logit_exp);
    }
    return math.divide(normalized_logits, denom)
}

// Determine holds in the climb
async function getCircles(frames) {
    // [[x,y, color], .....]
    const placements = await fetch(placements_path).then(d => d.json());
    const roles = await fetch(roles_path).then(d => d.json());
    let circles = [];
      for (let i=0; i < frames.length; i+=2) {
        let placement_id = String(frames[i].substring(1, frames[i].length+1));
        let res1 = placements[placement_id];
        let role_id = String(frames[i+1].substring(1, frames[i+1].length+1));
        let res2 = roles[role_id];
        let color, x, y;
        try {
            color = "#" + res2['color'];
            x = String(parseInt(res1['x']));
            y = String(parseInt(res1['y']));
        } catch (err) {
            exit(0)
        }
        let circle_data = [x, y, color];
        circles.push(circle_data);
      }
    return circles
}

// Get layout board specs
async function getLayoutInfo(layout) {
    // [layoutName, width, height, edge_left, edge_right, edge_bottom, edge_top]
    const layoutInfo = await fetch(layoutInfo_path).then(d => d.json());
    let layoutName = layout.replace(/\s+/g, '');
    return layoutInfo[layoutName]
}

// Inference function
export async function Inference(src, tgt, temp, samplingMethod, k=5, p=0.9, model, validationType) {
    // Target Sequence Length
    const seq_len = tgt.length;

    // Get Ids from the tokens
    const model_input = await getIds(src, tgt);

    // Determine which model to run
    let model_url;
    if (model === "tiny") {
        model_url = model_tiny_url;
    } else if (model === "large") {
        model_url = model_large_url;
    }

    // Run the session
    const session = await ort.InferenceSession.create(model_url, options);
    const output = await session.run(model_input, options);

    // Get Last Token logits and scale with temperature
    const logits = Array.from(output['output'].data);
    const lastTokenLogits = logits.slice((seq_len - 1) * vocab_size, seq_len * vocab_size);
    const scaled_logits = math.divide(lastTokenLogits, temp);

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
    
    // Conver token to ID
    let next_token = await getTokens(next_token_id);

    // Add token to existing Target Sequence
    let new_tgt = tgt.concat(next_token);

    // Autoregressively generate tokens until EOS token
    if (next_token[0] !== "<EOS>") {
        return await Inference(src, new_tgt, temp, samplingMethod, k, p, model, validationType)
    }
    
    // Final target sequence, with SOS and EOS tokens removed
    const final_tgt = new_tgt.slice(1, new_tgt.length-1)

    // Basic Validation
    const initalValidation = initalOutputValidation(final_tgt);

    // If basic validation is passed
    if (initalValidation) {
        // Get cirlces and layout information
        const circles = await getCircles(final_tgt);
        const layoutInfo = await getLayoutInfo(src[0]);
        // Strict Validation
        if (validationType === "strict") {
            const strictValidation = strictOutputValidation(circles, layoutInfo);
            // If strict Validation is not passed, regenerate climb from start
            if (!strictValidation) {
                regen += 1;
                return await Inference(src, ['<SOS>'], temp, samplingMethod, k, p, model, validationType)
            }
        }
        
        // Return climb information if valid
        return [final_tgt, circles, layoutInfo, regen]
    }
    
    // Regerate climb from start if generated one is invalid
    regen += 1;
    return await Inference(src, ['<SOS>'], temp, samplingMethod, k, p, model, validationType)
}