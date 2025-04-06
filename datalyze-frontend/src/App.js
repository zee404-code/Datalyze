import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './styles.css';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const App = () => {
  const [showOptions, setShowOptions] = useState(false);
  const [showUploadPage, setShowUploadPage] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [file, setFile] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState("");
  const optionsRef = useRef(null);
  const [status, setStatus] = useState("");
  const [downloadUrl, setDownloadUrl] = useState("");

  const handleGetStartedClick = () => {
    setShowOptions(true);
  };

  const handleBackButtonClick = () => {
    setShowOptions(false);
  };

  const handleOptionClick = async (option) => {
    const payload = {
      role: option === 'Business Owner' ? 'BO' : 'DA',
    };

    try {
      const response = await axios.post(`${API_BASE_URL}/submit-role`, payload);
      if (response.status === 200) {
        setShowOptions(false);
        setShowUploadPage(true);
      }
    } catch (error) {
      console.error('Error sending role to backend:', error);
      alert('Something went wrong. Please try again.');
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleUploadClick = async () => {
    if (!file) {
      setUploadStatus('Please select a file first.');
      return;
    }

    const fileType = file.type;
    if (
      ![
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'text/csv',
      ].includes(fileType)
    ) {
      setUploadStatus('Invalid file type. Only .xlsx, .xls, or .csv files are allowed.');
      return;
    }

    setUploadStatus('Uploading...');
    setUploadedFileName(file.name);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      if (response.status === 200) {
        setUploadStatus('File uploaded!');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setUploadStatus(`Error uploading file: ${error.message || String(error)}`);
    }
  };

  const handleViewInsightsClick = () => {
    if (!uploadedFileName) return;

    setIsProcessing(true);
    setProgress(0);
    setCurrentStep("⏳ Starting analysis...");

    const eventSource = new EventSource(`${API_BASE_URL}/get-insights-progress/${uploadedFileName}`);

    eventSource.onmessage = function (event) {
      try {
        const data = JSON.parse(event.data);

        if (data.progress) {
          setProgress(data.progress);
        }

        if (data.status) {
          setCurrentStep(data.status);
        }

        if (data.progress === 100 && data.filename) {
          const downloadUrl = `${API_BASE_URL}/download-report/${data.filename}`;

          const a = document.createElement('a');
          a.href = downloadUrl;
          a.download = 'Business_Intelligence_Report.pdf';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);

          eventSource.close();
          setIsProcessing(false);
          setCurrentStep("✅ Download should begin automatically.");
        }
      } catch (err) {
        console.error("Error parsing SSE message:", err);
        setCurrentStep("❌ Error during processing.");
        eventSource.close();
        setIsProcessing(false);
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE connection error:", err);
      setCurrentStep("❌ Connection error.");
      eventSource.close();
      setIsProcessing(false);
    };
  };

  useEffect(() => {
    if (showOptions && optionsRef.current) {
      optionsRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [showOptions]);

  return (
    <div className="content">
      {!showOptions && !showUploadPage ? (
        <>
          <h1>DATALYZE</h1>
          <div className="image-container">
            <img src="/assets/lyza_logo.png" alt="Lyza AI Consultant" className="image" />
          </div>
          <p className="description">
            MEET <strong>LYZA</strong> - YOUR AI POWERED BI CONSULTANT
          </p>
          <p className="tagline">WE CLEAN & ANALYZE YOUR DATA IN MINUTES</p>
          <button id="getStarted" onClick={handleGetStartedClick}>
            LET’S GET STARTED!
          </button>
        </>
      ) : showOptions ? (
        <div ref={optionsRef} className="new-options">
          <h2>CHOOSE HOW YOU WISH TO SEE YOUR INSIGHTS</h2>
          <div className="option-tile" onClick={() => handleOptionClick('Business Owner')}>
            <p className="option-text">
              <strong>Business Owner</strong>
              <br />
              Simple, Actionable Insights
            </p>
          </div>
          <div className="option-tile" onClick={() => handleOptionClick('Data Analyst')}>
            <p className="option-text">
              <strong>Data Analyst</strong>
              <br />
              Statistical + Analytical Insights
            </p>
          </div>
          <button id="backButton" onClick={handleBackButtonClick}>
            Back to Main Screen
          </button>
        </div>
      ) : showUploadPage ? (
        <div className="container">
          <h2>Upload Your File</h2>
          <p>Only Excel (.xlsx, .xls) or CSV files are allowed.</p>
          <input type="file" id="fileInput" accept=".xlsx,.xls,.csv" onChange={handleFileChange} />
          <button id="uploadButton" onClick={handleUploadClick}>Upload</button>
          <p id="uploadStatus">{uploadStatus}</p>

          {uploadStatus === 'File uploaded!' && !isProcessing && (
            <button id="insightButton" onClick={handleViewInsightsClick}>
              View Insights
            </button>
          )}

          {progress > 0 && (
            <div style={{ marginTop: "20px" }}>
              <div style={{ background: "#eee", borderRadius: "10px", height: "20px", width: "100%" }}>
                <div
                  style={{
                    width: `${progress}%`,
                    backgroundColor: "#4CAF50",
                    height: "100%",
                    borderRadius: "10px",
                    transition: "width 0.3s ease",
                    textAlign: "center",
                    color: "black",
                    fontWeight: "bold",
                    fontSize: "1.3rem"
                  }}
                >
                  {progress}%
                </div>
              </div>
              <p style={{ marginTop: "10px" }}>{status}</p>
            </div>
          )}

          <div className="image-container">
            <img src="/assets/hiw.png" alt="Lyza AI Consultant" className="image" />
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default App;
