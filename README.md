# Video Annotation System for Physical Activity Analysis

**Authors:**
- Alejandro Amu García  
- Santiago Escobar Leon  
- David Donneys  

---

## 🧑‍💻 Project Description

This project implements a system for analyzing physical activities using computer vision. It can identify five specific activities (walking towards the camera, walking away from the camera, turning, sitting, and standing) and perform real-time posture pattern analysis.

The system uses MediaPipe to extract body landmarks from video, processes this data to extract relevant features, and applies machine learning algorithms to classify activities and analyze movement patterns.

---

## 🚀 Key Features

- 📹 **Real-time physical activity detection:** Identification of activities such as walking, turning, sitting, and standing.
- 🧠 **Joint angle calculation:** Analysis of knees, hips, and torso angles.
- 📊 **Posture pattern analysis:** Evaluation of movements and postures.
- 🖥️ **Graphical interface:** Real-time visualization of results.
- 🎥 **Activity recording:** Data generation for training purposes.
- ⚡ **Optimized processing:** Efficient execution in real-time.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/santiesleo/AI-MotionClassifier
   cd AI-MotionClassifier

2. **Create and activate a virtual environment:**

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies:**

    pip install -r requirements.txt

## 🛠️ Project Execution

### 1. Video Capture
Run the `src/data_collection.py` script to capture videos of physical activities.

**Steps:**
- Start the script.
    `python src/data_collection.py`
- Select the subject by pressing `S` and entering the subject number.
- Choose the activity by pressing a key between 1 and 5:
  - `1`: Walking towards the camera.
  - `2`: Walking away from the camera.
  - `3`: Turning.
  - `4`: Sitting.
  - `5`: Standing.
- Press `R` to start or stop recording.
- Press `Q` to exit the system.

➡️ Captured videos will be saved in the `dataset_raw/` directory.

---

### 2. Video Processing
Use `video_processor.py` to extract landmarks and calculate joint angles.

**Steps:**
- Run the script.
    `python src/video_processor.py`
- The system will automatically process new videos in `dataset_raw/`.
- Results will be saved in JSON format in `dataset_processed/`.

---

### 3. Feature Extraction
Run `feature_extractor.py` to calculate relevant features (e.g., joint angles, velocities).

**Steps:**
- Execute the script.
    `python src/feature_extractor.py`
- Features will be saved in `features.csv` inside `dataset_processed/`.

---

### 4. Model Training
Use `model_training.py` to train activity classification models.

**Steps:**
- Run the script.
    `python src/model_training.py`
- The system will train and evaluate multiple models.
- Results will be saved in `dataset_processed/`.

---

### 5. Result Visualization
Use `visualization.py` to generate feature distribution graphs and impact diagrams.

**Steps:**
- Run the script.
    `python src/visualization.py`
- Graphs will be saved in `dataset_processed/visualizations/`.

---

### 6. Graphical Interface
Use `gui.py` for an interactive interface.

**Steps:**
- Run the script.
    `python app/gui.py`
- Load videos, visualize results, and perform analyses through the interface.

---

## 📂 File Descriptions

| File                | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `data_collection.py`| Enables video capture of physical activities. Includes functionalities for selecting subjects and activities, starting and stopping recordings. |
| `video_processor.py`| Processes videos to extract body landmarks and joint angles. Saves results in JSON format. |
| `feature_extractor.py` | Calculates relevant features such as joint angles and velocities. Generates a CSV file with extracted features.              |
| `model_training.py` | Trains machine learning models for classification. Evaluates model performance and saves the best-trained model. |
| `visualization.py`  | Generates feature distribution graphs and impact diagrams. Includes functions to visualize confusion matrices and compare models. |
| `gui.py`            | Provides a graphical interface to interact with the system. Allows users to load videos, visualize results, and perform analyses             |

---

## System Diagrams

## System Architecture

![Captura de pantalla 2025-06-06 224301](https://github.com/user-attachments/assets/23af100d-9965-4ce7-9a03-cf843fb9ae6b)

## Real-Time Detection PipeLine

![Captura de pantalla 2025-06-06 224335](https://github.com/user-attachments/assets/e92b2daa-962a-4cbc-a17e-73ab897e0820)

## Data Processing Pipeline

![Captura de pantalla 2025-06-06 224351](https://github.com/user-attachments/assets/f4a8b10d-36d8-4735-a619-6b3d2e7d623d)



## 📈 Conclusions and Reflection

The project successfully implemented a comprehensive system for analyzing physical activities, addressing challenges such as:

- ✅ **Efficient landmark processing**: The system ensures real-time performance while maintaining accuracy in feature extraction.
- ✅ **Robust model training**: Despite the constraints, the system achieves reliable classification results.
- ✅ **Clear result visualization**: The graphical interface and visualization scripts provide intuitive insights for users.

### Key Achievements
- The integration of MediaPipe for landmark extraction and machine learning for activity classification demonstrates the system's robustness.
- The modular design allows for easy scalability and adaptability to new activities or datasets.

### Future Improvements
- ➕ **Expand activity set for broader recognition:** Incorporate additional physical activities to broaden the system's applicability.
- 📊 **Enhance model accuracy with larger datasets and advanced techniques:** Use larger datasets and advanced machine learning techniques for improved classification.
- ⚡ **Improve real-time processing performance:** Further refine processing algorithms to reduce latency.

> This project represents a significant step forward in leveraging computer vision and machine learning for physical activity analysis, offering practical applications in health, sports, and rehabilitation domains.

---
