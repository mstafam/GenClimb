import { useState, useEffect } from 'react';
import { Inference } from './inference.js';
import * as React from 'react';
import CircularProgress from '@mui/material/CircularProgress';
import Climb from './Climb.js';

function App() {
  // States
  const [SelectedBoard, setSelectedBoard] = useState("");
  const [SelectedLayout, setSelectedLayout] = useState("");
  const [SelectedDifficulty, setSelectedDifficulty] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [frames, setFrames] = useState(null);
  const [latency, setLatency] = useState("");
  const [regenCount, setRegenCount] = useState("");
  const [circles, setCircles] = useState(null);
  const [layoutInfo, setlayoutInfo] = useState(null);
  const [temperature, setTemperature] = useState(1);
  const [validationType, setValidationType] = useState('basic');
  const [selectedModel, setSelectedModel] = useState('tiny');
  const [samplingMethod, setSamplingMethod] = useState('Multinomial');
  const [topK, setTopK] = useState(50);
  const [topP, setTopP] = useState(0.9);
  const [isWebGPUSupported, setIsWebGPUSupported] = useState(null);
  const [showNotice, setShowNotice] = useState(true);

  // Check for browser WebGPU support 
  useEffect(() => {
    const checkWebGPUSupport = async () => {
      if (!navigator.gpu) {
        setIsWebGPUSupported(false);
        return
      }
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          setIsWebGPUSupported(false);
          return
        }
        setIsWebGPUSupported(true);
      } catch (e) {
        setIsWebGPUSupported(false);
      }
    };

    checkWebGPUSupport();
  }, []);

  
  const changeSelectedBoard = (event) => {
    setSelectedBoard(event.target.value);
    setFrames(null);
  }

  const changeSelectedLayout = (event) => {
    setSelectedLayout(event.target.value);
    setFrames(null);
  }
  
  const changeSelectedDifficulty = (event) => {
    setSelectedDifficulty(event.target.value);
    setFrames(null);
  }

  const handleTemperatureChange = (event) => {
    setTemperature(event.target.value);
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleValidationChange = (event) => {
    setValidationType(event.target.value);
  };

  const handleSamplingMethodChange = (event) => {
    setSamplingMethod(event.target.value);
  };

  const handleTopKChange = (event) => {
    setTopK(parseInt(event.target.value));
  };

  const handleTopPChange = (event) => {
    setTopP(parseFloat(event.target.value));
  };

  // Board, Layout, and Difficulty Options
  const boards = ["Kilter Board Original", "Kilter Board Home"]
  const originalLayouts = ["8 x 12 Home", "7 x 10 Small", "12 x 14 Commerical", "12 x 12 with kickboard Square", "12 x 12 without kickboard Square", "16 x 12 Super Wide"]
  const homeLayouts = ["7x10 Full Ride LED Kit", "7x10 Mainline LED Kit", "7x10 Auxiliary LED Kit", "10x10 Full Ride LED Kit", "10x10 Mainline LED Kit", "10x12 Mainline LED Kit", "10x12 Full Ride LED Kit", "8x12 Mainline LED Kit", "8x12 Full Ride LED Kit", "10x10 Auxiliary LED Kit"]
  const difficulty = ["1a/V0", "1b/V0", "1c/V0", "2a/V0", "2b/V0", "2c/V0", "3a/V0", "3b/V0", "3c/V0", "4a/V0", "4b/V0", "4c/V0", "5a/V1", "5b/V1", "5c/V2", "6a/V3", "6a+/V3", "6b/V4", "6b+/V4", "6c/V5", "6c+/V5", "7a/V6", "7a+/V7", "7b/V8", "7b+/V8", "7c/V9", "7c+/V10", "8a/V11", "8a+/V12", "8b/V13", "8b+/V14", "8c/V15", "8c+/V16", "9a/V17", "9a+/V18", "9b/V19", "9b+/V20", "9c/V21", "9c+/V22"]

  let layout = originalLayouts.concat(homeLayouts);

  if (SelectedBoard === "Kilter Board Original") {
    layout = originalLayouts;
  } else if (SelectedBoard === "Kilter Board Home") {
    layout = homeLayouts;
  }
  
  let getOptions = (dropdown) => {
    if (dropdown) {
      return dropdown.map((el) => <option key={el}>{el}</option>)
    }
  }

  // Run inference and set the returned states
  const handleGenerate = async (event) => {
    event.preventDefault();
    if (!isWebGPUSupported) {
      alert("WebGPU is not supported in this browser. Please use a WebGPU-compatible browser to generate climbs.");
      return
    }
    setIsLoading(true);
    try {
      const start = performance.now();
      const [generatedFrames, generatedCircles, layoutInfo, regen] = await Inference([SelectedLayout, SelectedDifficulty], ['<SOS>'], temperature, samplingMethod, topK, topP, selectedModel, validationType); 
      const duration = (performance.now() - start).toFixed(1);
      setLatency(duration);
      setFrames(generatedFrames);
      setCircles(generatedCircles);
      setlayoutInfo(layoutInfo);
      setRegenCount(regen);
    } catch (error) {
      console.error("Error generating frames:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-900 p-4">
      <div className="max-w-7xl mx-auto">
      {showNotice && (
        <div className="bg-gray-700 text-gray-200 p-2 rounded-md mb-4 text-xs flex justify-between items-center">
          <span>For better results, enable Strict Validation in Advanced Options. Note: This may increase generation time.</span>
          <button 
            onClick={() => setShowNotice(false)} 
            className="ml-2 text-gray-400 hover:text-gray-200"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </button>
        </div>
      )}
        <div className="flex flex-col md:flex-row">
          {/* Left side - Input */}
          <div className="w-full md:w-1/2 p-6 bg-gray-800 border-r border-gray-700">
          <div className="text-sm text-center mb-4 text-gray-400">
          {isWebGPUSupported === null && "Checking WebGPU support..."}
          {isWebGPUSupported === false && (
            <>
            WebGPU is not supported in this browser. GenClimb works on most modern desktop browsers (Chrome, Edge, Safari on macOS), but generally not on mobile devices. 
            <a href="https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API#browser_compatibility" 
               className="text-blue-400 hover:text-blue-300 ml-1" 
               target="_blank" 
               rel="noopener noreferrer">
              See compatible browsers
            </a>.
          </>
          )}
          {isWebGPUSupported === true && "WebGPU is supported in this browser!"}
          </div>
            <h1 className="text-6xl md:text6xl text-center mb-6 font-extrabold bg-clip-text text-transparent bg-[linear-gradient(to_right,theme(colors.green.300),theme(colors.green.100),theme(colors.sky.400),theme(colors.yellow.200),theme(colors.sky.400),theme(colors.green.100),theme(colors.green.300))] bg-[length:200%_auto] animate-gradient">GENCLIMB</h1>
            <form onSubmit={handleGenerate} className="space-y-4">
              <select 
                onChange={changeSelectedBoard} 
                name="board" 
                value={SelectedBoard}
                className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white"
              >
                <option value="">Select Board</option>
                {getOptions(boards)}
              </select>
              <select 
                onChange={changeSelectedLayout} 
                name="layout" 
                value={SelectedLayout} 
                required
                className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white"
              >
                <option value="">Select Layout</option>
                {getOptions(layout)}
              </select>
              <select 
                onChange={changeSelectedDifficulty} 
                name="difficulty" 
                value={SelectedDifficulty} 
                required
                className="w-full p-2 bg-gray-700 border border-gray-600 rounded-md text-white"
              >
                <option value="">Select Difficulty</option>
                {getOptions(difficulty)}
              </select>
              <div className="mt-4">
      <details className="mb-4">
        <summary className="cursor-pointer text-sm text-gray-400">Advanced Options</summary>
        <div className="mt-2 space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Model:</span>
            <select 
              className="p-1 bg-gray-700 border border-gray-600 rounded-md text-sm text-white"
              value={selectedModel}
              onChange={handleModelChange}
            >
              <option value={"tiny"}>Tiny Quantized (Faster, Less Accurate)</option>
              <option value={"large"} disabled>Large Quantized (Slower, More Accurate)</option>
              <option disabled>Full-precision</option>
            </select>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Validation:</span>
            <select 
              className="p-1 bg-gray-700 border border-gray-600 rounded-md text-sm text-white"
              value={validationType}
              onChange={handleValidationChange}
            >
              <option value={"basic"}>Basic Validation</option>
              <option value={"strict"}>Strict Validition</option>
            </select>
          </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-400">Temperature:</span>
              <div className="flex items-center space-x-2 w-1/2">
              <input 
                type="range" 
                className="w-full" 
                min={0.1} 
                max={5} 
                step={0.1}
                value={temperature}
                onChange={handleTemperatureChange}
                list="temp"
              />
              <span className="text-sm text-gray-400 w-8 text-right">{temperature}</span>
            </div>
            <datalist id="temp">
              <option value="1"></option>
              <option value="2"></option>
              <option value="3"></option>
              <option value="4"></option>
              <option value="5"></option>
            </datalist>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-400">Sampling:</span>
            <div className="space-x-2">
              <select 
                className="p-1 bg-gray-700 border border-gray-600 rounded-md text-sm text-white"
                value={samplingMethod}
                onChange={handleSamplingMethodChange}
              >
                <option>Multinomial</option>
                <option>Top-k</option>
                <option>Top-p</option>
              </select>
              {samplingMethod === 'Top-k' && (
                <input 
                  type="number" 
                  className="p-1 bg-gray-700 border border-gray-600 rounded-md text-sm text-white w-16"
                  value={topK}
                  onChange={handleTopKChange}
                  min={1}
                  max={100}
                />
              )}
              {samplingMethod === 'Top-p' && (
                <input 
                  type="number" 
                  className="p-1 bg-gray-700 border border-gray-600 rounded-md text-sm text-white w-16"
                  value={topP}
                  onChange={handleTopPChange}
                  min={0}
                  max={1}
                  step={0.1}
                />
              )}
            </div>
          </div>
        </div>
      </details>
    </div>
              <button 
                type="submit"
                className={`w-full py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 ${
                  isWebGPUSupported
                    ? "bg-blue-600 text-white hover:bg-blue-700"
                    : "bg-gray-500 text-gray-300 cursor-not-allowed"
                }`}
                disabled={!isWebGPUSupported}
              >
                Generate
              </button>
                <div className="relative mt-2">
                <span className="text-gray-400 text-sm flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="group relative">
                    How it works
                    <span className="absolute left-0 bottom-full mb-2 hidden group-hover:block bg-gray-700 text-white text-xs p-2 rounded shadow-lg w-64">
                      GenClimb creates climbs right in your browser, using your device's GPU through WebGPU API. It's free to use, keeps your data private, and works best with modern browsers and graphics cards.
                    </span>
                  </span>
                </span>
              </div>
            </form>
          </div>
          
          {/* Right side - Climb/Loading */}
          <div className="w-full md:w-1/2 p-6 flex flex-col bg-gray-800">
            <div className="flex-grow flex items-center justify-center">
              <div className="w-full max-w-md aspect-w-3 aspect-h-4 bg-gray-700 rounded-lg overflow-hidden relative">
                <div className={`absolute inset-0 flex items-center justify-center ${isLoading ? 'opacity-100' : 'opacity-0'} transition-opacity duration-300`}>
                  <CircularProgress style={{ color: '#60A5FA' }} />
                </div>
                <div className={`${isLoading ? 'opacity-0' : 'opacity-100'} transition-opacity duration-300`}>
                  <Climb frames={frames} circles={circles} layoutInfo={layoutInfo}/>
                </div>


                {!isLoading && frames && <div className="p-2 mt-2 text-center w-full max-w-md aspect-w-3 aspect-h-4 bg-gray-700 rounded-lg overflow-hidden relative">
              <img src='./assets/holds.png' alt='holds-legend'></img>
            </div>}
              </div>
            </div>

            <div className="mt-6 flex justify-between items-center bg-gray-700 rounded-lg p-4 shadow-lg">
              <div className="text-sm text-gray-300">
                <span className="font-semibold">Inference Time:</span>
                <span className="ml-2">{latency ? `${latency} ms` : 'N/A'}</span>
              </div>
              <div className="text-sm text-gray-300">
                <span className="font-semibold">Refinement Iterations:</span>
                <span className="ml-2">{regenCount || '0'}</span>
              </div>
            </div>
          </div>
          </div>
        
        {/* Footer */}
        <footer className="bg-gray-900 px-6 py-4">
          <div className="max-w-7xl mx-auto flex flex-col sm:flex-row justify-between items-center text-sm text-gray-400">
            <span className="mb-2 sm:mb-0">
              Created by <a href="https://github.com/mstafam" className="text-blue-400 hover:text-blue-300 transition duration-150 ease-in-out">
                Mustafa
              </a>
            </span>
            <a href="https://github.com/mstafam/GenClimb" className="flex items-center hover:text-gray-200 transition duration-150 ease-in-out">
              <img src='./assets/github-logo.png' width={20} height={20} alt='GitHub logo' className="mr-2 invert" />
              <span>Source</span>
            </a>
          </div>
        </footer>
      </div>
    </main>
  );
}

export default App;
