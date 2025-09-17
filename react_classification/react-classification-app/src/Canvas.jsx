import React, { useRef, useState, useEffect } from "react";
import "./Canvas.css";

const Canvas = () => {
  const GRID_SIZE = 28;
  const GRID_SCALE = 15;
  const [mouseDown, setMouseDown] = useState(false);
  const [all_x, setAll_x] = useState([]);
  const [all_y, setAll_y] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [isClassifying, setIsClassifying] = useState(false);

  const canvasRef = useRef(null);

  useEffect(() => {
    const context = canvasRef.current.getContext("2d");
    context.fillStyle = "#000000";
    context.fillRect(0, 0, context.canvas.width, context.canvas.height);
  }, []);

  const get28x28PixelArray = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    const pixelArray = [];

    for (let row = 0; row < GRID_SIZE; row++) {
      const rowArray = [];
      for (let col = 0; col < GRID_SIZE; col++) {
        // Hole die Pixeldaten für dieses Grid-Feld
        const x = col * GRID_SCALE;
        const y = row * GRID_SCALE;

        // Nimm den mittleren Pixel des Grid-Feldes als Repräsentant
        const imageData = context.getImageData(
          x + Math.floor(GRID_SCALE / 2),
          y + Math.floor(GRID_SCALE / 2),
          1,
          1
        );

        const [r, g, b, a] = imageData.data;
        const gray = Math.round((r + g + b) / 3);
        rowArray.push(gray);
      }
      pixelArray.push(rowArray);
    }

    return pixelArray;
  };

  // Klassifizierungsfunktion
  const classifyDrawing = async () => {
    if (isClassifying || all_x.length === 0) return;

    setIsClassifying(true);
    try {
      const imageData = get28x28PixelArray();
      const normalizedData = imageData.flat().map((pixel) => pixel / 255.0);

      const response = await fetch("http://localhost:5001/classify", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ pixels: normalizedData }),
      });

      if (!response.ok) {
        throw new Error("Classification failed");
      }

      const result = await response.json();
      setPrediction({
        class: result.prediction,
        confidence: (result.confidence * 100).toFixed(1),
        probabilities: result.probabilities,
      });
    } catch (error) {
      console.error("Classification failed:", error);
      setPrediction({
        class: "Error",
        confidence: "0",
        error: error.message,
      });
    } finally {
      setIsClassifying(false);
    }
  };

  const handleMouseMove = (event) => {
    if (!mouseDown) return;

    // Optional: Implement drawing while moving the mouse
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // snap the pixel values to the grid
    const snappedX = Math.floor(x / GRID_SCALE);
    const snappedY = Math.floor(y / GRID_SCALE);

    setAll_x((prev) => [...prev, snappedX]);
    setAll_y((prev) => [...prev, snappedY]);

    const imageData = get28x28PixelArray();

    console.log(imageData);

    for (let i = 0; i < all_x.length; i++) {
      // paint a rect around the cursor position with a lighter color
      for (let di = -1; di <= 1; di++) {
        for (let dj = -1; dj <= 1; dj++) {
          if (
            (di === 1 && dj === -1) ||
            (di === -1 && dj === 1) ||
            (di === -1 && dj === -1) ||
            (di === 1 && dj === 1)
          )
            continue; // skip the center

          const x_pos = all_x[i] + di;
          const y_pos = all_y[i] + dj;

          if (imageData[y_pos][x_pos] !== 0) continue;

          const gray = Math.floor(195 - Math.random() * 100);
          const alpha = Math.max(1 - Math.random() * 0.7, 0);
          context.fillStyle = `rgba(${gray}, ${gray}, ${gray}, ${alpha})`;

          // ensure we don't go out of bounds
          if (x_pos < 0 || x_pos >= GRID_SIZE) continue;
          if (y_pos < 0 || y_pos >= GRID_SIZE) continue;

          context.beginPath();
          context.rect(
            x_pos * GRID_SCALE,
            y_pos * GRID_SCALE,
            GRID_SCALE,
            GRID_SCALE
          );
          context.fill();
        }
      }
    }

    const mainGray = Math.floor(255 - Math.random() * 40);
    const mainAlpha = Math.max(1 - Math.random() * 0.2, 0);
    context.fillStyle = `rgba(${mainGray}, ${mainGray}, ${mainGray}, ${mainAlpha})`;
    for (let i = 0; i < all_x.length; i++) {
      if (imageData[all_y[i]][all_x[i]] > 200) continue;
      context.beginPath();
      context.rect(
        all_x[i] * GRID_SCALE,
        all_y[i] * GRID_SCALE,
        GRID_SCALE,
        GRID_SCALE
      );
      context.fill();
    }

    classifyDrawing();
  };

  const clearCanvas = () => {
    const context = canvasRef.current.getContext("2d");
    context.fillStyle = "#000000";
    context.fillRect(0, 0, context.canvas.width, context.canvas.height);
    setAll_x([]);
    setAll_y([]);
    setPrediction(null);
  };

  return (
    <div className="canvas-container">
      <canvas
        ref={canvasRef}
        className="canvas"
        width={GRID_SIZE * GRID_SCALE}
        height={GRID_SIZE * GRID_SCALE}
        onMouseDown={() => setMouseDown(true)}
        onMouseUp={() => setMouseDown(false)}
        onMouseMove={handleMouseMove}
        style={{ border: "2px solid #282828ff", cursor: "crosshair" }}
      />

      <div className="controls">
        <button className="clear-canvas-button" onClick={clearCanvas}>
          Clear Canvas
        </button>
      </div>

      <div className="prediction">
        {prediction && (
          <div>
            <h2>Prediction: {prediction.class}</h2>
          </div>
        )}
      </div>
    </div>
  );
};

export default Canvas;
