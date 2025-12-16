Traffic Sign Detection with YOLOv8
This project focuses on building and evaluating a traffic sign detection system using the YOLOv8 object detection framework. The goal is to accurately identify various traffic signs, including green lights, red lights, stop signs, and speed limit signs, which is crucial for autonomous driving applications.

Dataset
The dataset used for this project is sourced from Kaggle: pkdarabi/cardetection. It contains images of various traffic signs with corresponding YOLO format annotations.

Dataset Distribution
The distribution of classes across the training, validation, and test sets is as follows:

                    Class  Train  Valid  Test
              Green Light    542    122   110
                Red Light    585    108    94
           Speed Limit 10     19      0     3
          Speed Limit 100    267     52    46
          Speed Limit 110    101     17    21
          Speed Limit 120    252     60    44
           Speed Limit 20    285     56    46
           Speed Limit 30    334     74    60
           Speed Limit 40    235     55    53
           Speed Limit 50    283     71    50
           Speed Limit 60    301     76    45
           Speed Limit 70    318     78    53
           Speed Limit 80    323     56    61
           Speed Limit 90    168     38    34
                     Stop    285     81    50
Models and Experiments
Two YOLOv8 models (yolov8n and yolov8m) were trained and evaluated, followed by a fine-tuning phase and hyperparameter tuning experiments.

1. Baseline Training (YOLOv8n and YOLOv8m)
Initial training runs were performed for 20 epochs using yolov8n.pt and yolov8m.pt pretrained weights on the COCO dataset.

Comparison of Baseline Models:

                        model  precision    recall     mAP50  mAP50-95
              0  yolov8n_baseline   0.947907  0.888775  0.956316  0.826207
              1  yolov8m_baseline   0.953047  0.922588  0.968839  0.842941
YOLOv8m showed superior performance, which was expected due to its larger capacity.

2. Fine-tuning YOLOv8m
The yolov8m model was further fine-tuned for an additional 60 epochs with a slightly reduced learning rate (lr0=0.005) and early stopping patience (15 epochs).

Baseline vs. Fine-tuned YOLOv8m Metrics:

                 Metric  Baseline  Finetune
                  mAP50   0.96833   0.97538
               mAP50-95   0.84098   0.85369
              Precision   0.95271   0.97284
                 Recall   0.92289   0.94167
Fine-tuning significantly improved all key metrics, especially mAP50-95.

Training Speed Comparison:

                 Epochs  Avg Time per Epoch (sec)  Total Training Time (sec)
          20 (Baseline)                300.248490                  6004.9698
          60 (Finetune)                856.089148                 51365.3489
3. Image Size Experiment
The impact of input image size (imgsz) on model performance and inference speed was evaluated using yolov8m trained for 40 epochs.

Accuracy Comparison (by Image Size):

           imgsz  last_epoch  last_mAP50  last_mAP50-95  last_precision  last_recall  best_epoch  best_mAP50-95
             512          40     0.96381        0.84283         0.95593      0.92132          40        0.84283
             640          40     0.96815        0.84559         0.96546      0.93925          40        0.84559
             800          40     0.97433        0.85413         0.96757      0.95717          39        0.85597
Inference Speed Comparison:

           imgsz  images_tested  total_time_sec  avg_time_sec_per_image       fps
             512             50        0.519174                0.010383 96.306852
             640             50        0.598983                0.011980 83.474816
             800             50        0.724638                0.014493 68.999971
Larger image sizes generally lead to higher accuracy but lower FPS.

4. Learning Rate Sweep
An experiment was conducted to find the optimal initial learning rate (lr0) for training YOLOv8m models. Three learning rates (0.003, 0.005, 0.01) were tested with models trained for 40 epochs and imgsz=640.

Learning Rate Sweep Comparison:

            lr0  last_precision  last_recall  last_mAP50  last_mAP50-95  best_epoch  best_mAP50-95
          0.003         0.96546      0.93925     0.96815        0.84559          40        0.84559
          0.005         0.96546      0.93925     0.96815        0.84559          40        0.84559
          0.010         0.96546      0.93925     0.96815        0.84559          40        0.84559
Note: It appears that due to how the lr0 was set (optimizer=auto), the actual effective learning rate might have been similar across these experiments, leading to identical results in the final epoch. Further investigation or manual LR scheduling might be needed for a more distinct comparison.

Getting Started
To replicate the experiments or use the trained models:

Clone the repository:

          git clone <your_repo_url>
          cd <your_repo_name>
Setup Environment (Colab/Python): Ensure you have ultralytics installed:

          !pip install ultralytics
Download the Dataset: Follow the steps in the notebook to download the pkdarabi/cardetection dataset from Kaggle.

Run Training/Evaluation: Refer to the provided Jupyter notebook (.ipynb file) for detailed code on training, fine-tuning, and evaluating the models.
