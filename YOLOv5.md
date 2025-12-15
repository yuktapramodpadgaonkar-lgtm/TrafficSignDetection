````markdown
# Traffic Signal & Road Sign Detection with YOLOv5

Robust real-time detection of traffic lights and road signs using a custom-trained YOLOv5 model on a Kaggle traffic dataset.

---

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)
![YOLOv5](https://img.shields.io/badge/Model-YOLOv5s-green.svg)
![Status](https://img.shields.io/badge/Status-Research%20Project-brightgreen.svg)

## 1. Overview

This project trains a **YOLOv5** object detection model to recognize **traffic lights and road signs** from images.  

The pipeline is implemented in a single Jupyter notebook and covers the full workflow:

- Dataset download from Kaggle  
- Data preparation and sanity checks  
- Visual inspection of labels and bounding boxes  
- Class distribution analysis  
- Training a YOLOv5s model  
- Evaluation on a held-out test set  
- Inference on sample traffic images  
- Simple inference speed benchmark  

The notebook is designed to work smoothly on **Google Colab** as well as a local machine with GPU support.

---

## 2. Dataset

**Source**

- Kaggle dataset: `pkdarabi/cardetection`  
- After download, the dataset is unpacked into a folder called `traffic_dataset/`.

**Structure (after preprocessing)**

The notebook expects the dataset to be in the standard YOLO layout:

```text
traffic_dataset/
├── train
│   ├── images/
│   └── labels/
├── valid
│   ├── images/
│   └── labels/
└── test
    ├── images/
    └── labels/
````

**Classes (15 total)**

```text
0: Green Light
1: Red Light
2: Speed Limit 10
3: Speed Limit 100
4: Speed Limit 110
5: Speed Limit 120
6: Speed Limit 20
7: Speed Limit 30
8: Speed Limit 40
9: Speed Limit 50
10: Speed Limit 60
11: Speed Limit 70
12: Speed Limit 80
13: Speed Limit 90
14: Stop
```

Annotation format follows the standard **YOLO text format**:

```text
<class_id> <x_center> <y_center> <width> <height>
```

where all coordinates are normalized to `[0, 1]`.

---

## 3. Project Structure

Once you run the notebook, your workspace will look roughly like this:

```text
.
├── YOLOv5_v5.ipynb
├── yolov5/                       # Cloned Ultralytics YOLOv5 repo
├── traffic_dataset/              # Kaggle dataset (train/valid/test)
├── traffic_data.yaml             # Custom data config for YOLOv5
├── sample_visualizations.png     # Sample images with bounding boxes
├── class_distribution.png        # Class distribution across splits
├── test_predictions.png          # YOLOv5 predictions on test images
└── runs/
    ├── train/
    │   └── traffic_yolov5s_exp/  # Training runs, weights, metrics
    └── test/
        └── traffic_eval/         # Evaluation results and plots
```

You can commit the notebook and selected output images (not the entire `runs/` folder if it’s too large).

---

## 4. Setup & Installation

### 4.1. Prerequisites

* Python **3.8+**
* Recommended: **GPU with CUDA** support
* A Kaggle account and a valid `kaggle.json` API key

### 4.2. Create and Activate Environment (Optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 4.3. Install Dependencies

The notebook installs all required packages with:

```python
!pip install -q torch torchvision torchaudio
!pip install -q opencv-python matplotlib seaborn pandas pillow pyyaml
!pip install -q kaggle
```

You can also install them manually:

```bash
pip install torch torchvision torchaudio
pip install opencv-python matplotlib seaborn pandas pillow pyyaml kaggle
```

---

## 5. Downloading the Dataset (Kaggle)

The notebook uses the Kaggle API:

1. Create a Kaggle API token from your Kaggle account.
2. Download **`kaggle.json`**.
3. Upload `kaggle.json` when prompted in the notebook (Colab flow), or place it manually at:

```text
~/.kaggle/kaggle.json
```

The dataset is then downloaded and extracted with:

```bash
kaggle datasets download -d pkdarabi/cardetection -p traffic_dataset --unzip
```

The notebook also includes sanity checks to locate the dataset folder and verify:

* Number of images in **train/valid/test**
* Number of label files in each split

---

## 6. Data Exploration & Visualization

### 6.1. Sample Images with Bounding Boxes

The function `visualize_samples()`:

* Picks a few training images
* Reads the corresponding YOLO label files
* Draws bounding boxes and class names using OpenCV
* Saves the resulting grid to `sample_visualizations.png`

You can quickly confirm:

* Bounding boxes are aligned correctly
* Classes look reasonable (e.g., speed limits, red/green lights, stop signs)

### 6.2. Class Distribution

The function `analyze_class_distribution()`:

* Iterates over label files in `train/`, `valid/`, and `test/`
* Counts how many instances of each class appear in each split
* Builds a Pandas DataFrame and produces:

  * A bar chart for per-class counts across splits
  * A pie chart showing the training set class distribution
* Saves the plot as `class_distribution.png`
* Prints total instances and number of classes seen per split

This helps identify **class imbalance** early on.

---

## 7. YOLOv5 Configuration

A custom `traffic_data.yaml` is created automatically with:

```yaml
path: /absolute/path/to/traffic_dataset
train: train/images
val: valid/images
test: test/images
nc: 15
names:
  - Green Light
  - Red Light
  - Speed Limit 10
  - Speed Limit 100
  - Speed Limit 110
  - Speed Limit 120
  - Speed Limit 20
  - Speed Limit 30
  - Speed Limit 40
  - Speed Limit 50
  - Speed Limit 60
  - Speed Limit 70
  - Speed Limit 80
  - Speed Limit 90
  - Stop
```

The notebook also detects:

* PyTorch version
* Whether CUDA is available
* GPU name and CUDA version (if any)

---

## 8. Training the Model

The training cell launches YOLOv5 training with:

```bash
python yolov5/train.py \
    --img 640 \
    --batch 16 \
    --epochs 20 \
    --data traffic_data.yaml \
    --weights yolov5s.pt \
    --project runs/train \
    --name traffic_yolov5s_exp \
    --patience 5 \
    --cache
```

Key choices:

* **Model**: `yolov5s.pt` (small variant, fast and lightweight)
* **Image size**: `640 × 640`
* **Batch size**: `16`
* **Epochs**: `20` (for quick experimentation; you can increase to 40–60 for better performance)
* **Early stopping**: `patience=5` (stops if no improvement for 5 epochs)
* **Cache**: speeds up training by caching images

Training time is measured and printed at the end of this stage.

---

## 9. Training Metrics & Plots

After training, the notebook reads `runs/train/traffic_yolov5s_exp/results.csv` and builds a dashboard of plots:

* Train vs. validation **box loss**
* Train vs. validation **objectness loss**
* Train vs. validation **classification loss**
* **mAP@0.5**
* **Precision**
* **Recall**

These are saved as a combined figure: `training_metrics.png`.

It also prints the final values from the last epoch, including:

* `mAP@0.5`
* `mAP@0.5:0.95`
* `precision`
* `recall`

> In your README, you can add the actual numbers from your best run here once you have them.

---

## 10. Evaluation on Test Set

The best weights from training are loaded from:

```text
runs/train/traffic_yolov5s_exp/weights/best.pt
```

The notebook then runs YOLOv5’s built-in evaluation on the **test split**:

```bash
python yolov5/val.py \
    --data traffic_data.yaml \
    --weights runs/train/traffic_yolov5s_exp/weights/best.pt \
    --img 640 \
    --batch 16 \
    --task test \
    --save-txt \
    --save-conf \
    --project runs/test \
    --name traffic_eval
```

Evaluation outputs:

* Confusion matrix
* Precision–Recall curve
* F1 curve
* P and R curves
* A `results.csv` with test metrics

These plots are saved inside:

```text
runs/test/traffic_eval/
```

The notebook prints:

* Test **precision**
* Test **recall**
* **mAP@0.5**
* **mAP@0.5:0.95**

You can copy those numbers into a small results table in this section, for example:

```text
Metric           Value
--------------   -----
mAP@0.5          <fill from results.csv>
mAP@0.5:0.95     <fill from results.csv>
Precision        <fill from results.csv>
Recall           <fill from results.csv>
```

---

## 11. Inference on Sample Images

For inference, the notebook loads the trained model via `torch.hub`:

```python
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='runs/train/traffic_yolov5s_exp/weights/best.pt')
model.conf = 0.5   # confidence threshold
model.iou = 0.45   # IoU threshold for NMS
```

It then:

* Takes a subset of images from `test/images`
* Runs inference
* Renders predictions (boxes + labels) using YOLOv5’s `.render()` utility
* Arranges them in a 3 × 3 grid
* Saves the final figure as `test_predictions.png`

You can embed a sample image in your README after you commit it, for example:

```markdown
### Sample Predictions

![Test predictions](test_predictions.png)
```

---

## 12. Inference Speed Benchmark

An optional section of the notebook measures approximate inference latency:

* Warms up the model for a few iterations
* Runs inference 100 times on a single test image
* Reports:

  * Mean latency (ms / image)
  * Standard deviation
  * Min / Max latency
  * Approximate FPS (`1000 / mean_ms`)

You can summarize the result in the README like:

```text
Hardware: <your GPU or CPU>
Average inference time: ~XX ms/image
Approximate throughput: YY FPS
```

(Replace `XX` and `YY` using your own run.)

---

## 13. How to Use This Project

### Option A – Run the Notebook Directly

1. Open `YOLOv5_v5.ipynb` in **Google Colab** or Jupyter.
2. Run cells **top to bottom**:

   * Install dependencies
   * Set up Kaggle and download dataset
   * Verify dataset
   * Visualize samples
   * Analyze class distribution
   * Train YOLOv5
   * Evaluate on test set
   * Run inference
   * (Optional) Run speed benchmark
3. Inspect the generated images and metrics in `runs/` and the project root.

### Option B – Reuse the Trained Weights

If you only want inference:

1. Place `best.pt` from `runs/train/traffic_yolov5s_exp/weights/` in a known location.
2. Load the model:

```python
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt')
model.conf = 0.5
model.iou = 0.45

results = model('path/to/image.jpg')
results.show()   # or results.save()
```

---

## 14. Possible Improvements / Future Work

A few directions to extend this project:

* Train for **more epochs** (40–60) to squeeze out extra performance.
* Try larger models: **YOLOv5m** or **YOLOv5l** for higher accuracy.
* Experiment with:

  * Different input sizes (e.g., 512, 800)
  * Stronger augmentations
  * Class rebalancing techniques if the dataset is skewed.
* Evaluate on **real-world video** (dashcam footage) instead of still images.
* Export and deploy the model to:

  * Edge devices (NVIDIA Jetson, Raspberry Pi with NPU)
  * Mobile (ONNX / TensorRT / CoreML via conversion)
* Add a simple **web or streamlit demo** for interactive usage.

---

## 15. Acknowledgements

* **Ultralytics YOLOv5** for the detection framework.
* **Kaggle** and the dataset author (`pkdarabi`) for the cardetection dataset.
* Open-source Python ecosystem: PyTorch, OpenCV, Matplotlib, Seaborn, and Pandas.

---

## 16. License

Specify your license here, for example:

```text
This project is licensed under the MIT License. See the LICENSE file for details.
```

---

> **Note for reviewers / recruiters:**
> The notebook in this repository walks through the *full* lifecycle of an object detection project: data preparation, EDA, model training, evaluation, and inference. The code is organized so it can be used both as a reference implementation and as a starting point for production experiments.

```

---

If you’d like, I can also:

- Add a short **“Project Motivation”** paragraph tailored to your resume and MSAI program.
- Help you write a **one-sentence tagline** to put at the top of your GitHub profile that links to this project as your flagship DL work.
```
