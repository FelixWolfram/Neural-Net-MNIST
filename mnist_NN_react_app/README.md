# React Classification App

## Setup Instructions

### 1. Python Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the Python server
python server.py
```
The server will run on http://localhost:5000

### 2. React Frontend Setup
```bash
# Install Node dependencies (if not done)
npm install

# Start the React dev server
npm run dev
```
The frontend will run on http://localhost:5173

### 3. Replace Model Weights
Edit `server.py` and replace the dummy weights with your trained model:
```python
# Load your trained weights
weights1 = your_trained_weights1
biases1 = your_trained_biases1
weights2 = your_trained_weights2
biases2 = your_trained_biases2
```

### 4. Test the Setup
1. Start Python server: `python server.py`
2. Start React app: `npm run dev`
3. Draw on the canvas
4. Classification should happen in real-time

## Files Structure
```
/
├── src/
│   ├── Canvas.jsx     # Drawing canvas component
│   ├── App.jsx        # Main React app
│   └── ...
├── server.py          # Python Flask classification server
├── requirements.txt   # Python dependencies
└── package.json       # Node dependencies
```+ Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
