import "./App.css";
import Canvas from "./Canvas";

function App() {
  return (
    // the Provider of the context ensures, that all children have access to the provided data (stats, setStats)
    // you could say, the Context is just there for providing the values and defining the type, while the data itself is store in the useState
    <div className="app-container">
      <h1>React Classification App</h1>
      <Canvas></Canvas>
    </div>
  );
}

export default App;
