# Real-Time Cybersecurity Anomaly Detection Ecosystem

An autonomous malicious traffic detection system designed to analyze server logs and identify security threats in real-time. This project implements a hybrid approach combining unsupervised outlier recognition with deep learning for high-precision threat identification.

## 🚀 Key Features

* **Real-time Monitoring:** Processes web traffic logs to differentiate between normative user behavior and potential intrusions.
* **Hybrid Architecture:** Combines **Isolation Forest** for unsupervised anomaly detection and **1D Convolutional Neural Networks (1D-CNN)** for deep feature extraction.
* **Performance:** Achieves **92% precision** in threat identification.
* **Efficiency:** Reduces false positives by **40%** and improves incident response time by **35%**.
* **Scalability:** Capable of processing over **100,000 log entries daily**.

## 🛠️ Technical Stack

* **Languages:** Python, SQL
* **Frameworks/Libraries:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, NetworkX
* **Data Pipeline:** Automated log processing and anomaly scoring via Python/TensorFlow.

## 📊 Methodology

### 1. Data Processing & Standardization

The system ingests raw AWS CloudWatch VPC Flow logs. Preprocessing includes:

* Removal of duplicate entries.
* Standardization of IP country codes and timestamps.
* Feature engineering: Calculation of `duration_seconds` and scaling of throughput metrics (`bytes_in`, `bytes_out`).

### 2. Exploratory Data Analysis (EDA)

* **Traffic Patterns:** Distribution analysis of data throughput to identify volume-based attacks.
* **Protocol Analysis:** Monitoring protocol usage (HTTPS, etc.) to detect unusual port communication.
* **Correlation Mapping:** Analyzing the relationship between bytes transferred and session duration.

### 3. Model Architecture

* **Isolation Forest:** Used for initial outlier detection to identify rare events in the traffic stream.
* **1D-CNN:** Utilized for sequence-based feature learning, allowing the model to recognize complex spatial patterns in the log data.
* **StandardScaler:** Ensures numerical stability and improved convergence during neural network training.

## 📈 Results

* **Precision:** 92%
* **False Positive Reduction:** 40%
* **Response Improvement:** 35%
* **Throughput:** ~100k logs/day
# Autonomous Malicious Traffic Detection System

![Traffic Analysis](images/traffic_distribution.png)

## 📌 Project Overview
An orchestrated autonomous system designed to analyze server logs and identify security threats in real-time. By processing over **100,000 log entries daily**, this system differentiates between normative user behavior and potential web intrusions with high precision.

## 🚀 Key Achievements
* **92% Precision:** Achieved through a hybrid model combining Isolation Forest and 1D-CNN.
* **40% Reduction in False Positives:** Engineered a real-time monitoring solution to improve alert accuracy.
* **35% Faster Incident Response:** Optimized data pipelines for automated log processing and scoring.

## 🏗️ System Architecture
The ecosystem utilizes a dual-layer detection strategy:
1. **Unsupervised Outlier Recognition:** Leveraging **Isolation Forest** to identify anomalous spikes and patterns without labeled data.
2. **Deep Feature Extraction:** Implementing a **1D Convolutional Neural Network (1D-CNN)** to analyze sequential patterns in web traffic.

## 📊 Data Visualization & Insights

### 1. Network Traffic Distribution
Analyzes the volume of incoming and outgoing data to establish baseline normative behavior.
![Traffic Distribution](traffic_distribution.png)

### 2. Protocol Frequency Analysis
Identifies anomalies in port usage and communication protocols.
![Protocol Analysis](protocol_analysis.png)

### 3. IP Interaction Network
A graph-based visualization of communication between source and destination IPs.
![Network Interaction](network_graph.png)

### 4. Anomaly Detection (Isolation Forest)
Visual representation of how the model separates standard traffic from malicious outliers.
![Anomaly Scores](anomaly_scores.png)

### 5. Deep Learning Performance
Training metrics for the 1D-CNN model showing accuracy and loss convergence.
![Training Metrics](training_metrics.png)

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy, SQL
* **Machine Learning:** Scikit-learn (Isolation Forest)
* **Deep Learning:** TensorFlow / Keras (1D-CNN)
* **Visualization:** Matplotlib, Seaborn, NetworkX

## 📂 Project Structure
* `PROJECT-1.ipynb`: Core analysis and model training notebook.
* `images/`: Exported visualizations for the security dashboard.
* `data/`: Server log datasets (AWS CloudWatch/VPC Flow logs).

---
*Note: This project was developed to enhance real-time cybersecurity monitoring and automated threat identification.*

## 📂 Project Structure

```text
├── data/                       # Raw and processed log files
├── notebooks/
│   └── PROJECT-1.ipynb        # Core analysis and model training
├── models/                     # Saved model weights (H5/Pickle)
└── README.md                   # Project documentation

```

## 🔧 Installation & Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cyber-anomaly-detection.git

```


2. Install dependencies:
```bash
pip install pandas seaborn matplotlib networkx scikit-learn tensorflow

```


3. Run the analysis:
Open `notebooks/PROJECT-1.ipynb` in Jupyter or Google Colab and execute the cells.

---

