# GenClimb

Try it out at [genclimb.com](https://www.genclimb.com/)

GenClimb is a generative AI model designed to create climbing routes for Standardized Interactive Climbing Training Boards (SICTBs). Utilizing a seq2seq transformer architecture, GenClimb generates climbs conditioned on specific board layouts and difficulty levels.

**Currently, only Kilter Boards are supported.**

## Inferencing

GenClimb performs inferencing directly in your browser, using WebAssembly. This approach offers several benefits:

- **Privacy**: Your data stays on your device, with no information sent to external servers.
- **Performance**: Utilizes ONNX Runtime Web for efficient, on-device inference.
- **Edge Computing**: Brings AI directly to your device, eliminating the need for server-side processing.

## Model Details: GenClimb-Tiny-Quantized

The model is quantized to int8 for improved efficiency.

### Architecture Parameters
- Dimension: 256
- Number of Heads: 4
- Number of Layers: 2
- Feed-Forward Dimension: 256
- Dropout: 0.3
- Activation Function: GELU
- Layer Normalization Epsilon: 1e-5

### Training Configuration
- Device: CUDA
- Learning Rate: 1e-4
- Epochs: 8
- Weight Decay: 0.015
- Batch Size: 64
- Train/Test Split: 90/10

## Resources

- [GenClimb-Large-Quantized Model (Hugging Face)](https://huggingface.co/mstafam/genclimb-large-quantized)
- [Curated Kilter Board Dataset](https://huggingface.co/datasets/mstafam/Kilter-Board-Dataset)

## Acknowledgements

Special thanks to lemeryfertitta for the [BoardLib utility](https://github.com/lemeryfertitta/BoardLib), which was instrumental for getting the data.

## Future Improvements

1. Integration with Kilter Account to save climbs.
2. Improved inference efficiency through the use of KV caching and encoder caching.

## Run Locally

1. Clone this repository: 
```
git clone https://github.com/mstafam/GenClimb.git
```
2. Navigate to the project directory:
```
cd GenClimb
```
3. Install dependencies:
```
npm install
```
4. Start the application:
```
npm start
```

The app should launch at http://localhost:3000.
