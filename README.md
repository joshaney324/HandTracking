# HandTracking

A Python application that utilizes MediaPipe and OpenCV to perform real-time hand tracking using your webcam.

---

## Features

- **Real-Time Hand Tracking**: Detects and tracks hands in real-time using your webcam feed.
- **Landmark Detection**: Identifies 21 hand landmarks per detected hand.
- **Visualization**: Draws landmarks and connections on the video feed for visual feedback.
- **ASL Alphabet Detection**: Identifies the ASL Alphabet exluding J and Z.

---

## Tech Stack

- **Programming Language**: Python
- **Libraries**:
  - [MediaPipe](https://google.github.io/mediapipe/) for hand tracking.
  - [OpenCV](https://opencv.org/) for video capture and image processing.
  - [TensorFlow](https://www.tensorflow.org/api_docs) for model.

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/joshaney324/HandTracking.git
   cd HandTracking
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Make sure to install all required packages.

4. **Run the Application**:
   ```bash
   python src/hand_tracking.py
   ```

---

## Usage

- Upon running the application, a window will display the webcam feed with hand landmarks overlaid.
- Press `q` to exit the application.

---

## Project Structure

```
HandTracking/
├── data/
├── src/
│   └── base_hand_tracking/
|   └── ASL_alphabet_detection/
└── README.md
```

---



## Contact

**Josh Aney**  
GitHub: [@joshaney324](https://github.com/joshaney324)  
Email: [josh.aney@icloud.com](mailto:josh.aney@icloud.com)
