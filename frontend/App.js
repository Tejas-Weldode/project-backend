import React, { useState } from "react";
import axios from "axios";

function App() {
  const [sentence, setSentence] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setResult(null);
    setError(null);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        { sentence },
        { headers: { "Content-Type": "application/json" } }  // Ensure correct headers
      );
      setResult(response.data.quality);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || "An error occurred.");
    }
  };

  return (
    <div className="container">
      <h1>Text Quality Prediction</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={sentence}
          onChange={(e) => setSentence(e.target.value)}
          placeholder="Enter a sentence..."
          required
        />
        <button type="submit">Predict</button>
      </form>

      {result && <h3>Predicted Quality: {result}</h3>}
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
}

export default App;
