# 🎥 Lip Reading with Deep Learning  

A deep learning project that predicts spoken words from silent videos using **CNN + BiLSTM + CTC Loss**.  
Trained on the **GRID dataset** (20 speakers) and evaluated on unseen speakers, this project demonstrates end-to-end lip reading — from raw data preprocessing to model training, fine-tuning, evaluation, and deployment with a Streamlit UI.  

---

## 📂 Project Overview  

- **Dataset**: GRID Corpus (~15GB)  
  - 20 Speakers (S1–S20) → Training  
  - 5 Speakers (S21–S25) → Unseen Evaluation  
- **Preprocessing**:  
  - Extracted frames from videos using OpenCV & ffmpeg  
  - Stored aligned transcripts in JSON format  
  - Normalized and resized frames to **64×64**  
- **Model**:  
  - CNN for feature extraction  
  - BiLSTM for temporal modeling  
  - CTC Loss for alignment-free sequence prediction  

---

## ⚙️ Tech Stack  

- **Programming**: Python  
- **Deep Learning**: PyTorch  
- **Data Processing**: OpenCV, NumPy, Pandas  
- **Visualization**: Matplotlib, Seaborn  
- **Deployment**: Streamlit  
- **Others**: ffmpeg for frame extraction  

---

## 📊 Training  

- **Batch Size**: 4  
- **Optimizer**: Adam  
- **Learning Rate**: 1e-4  
- **Epochs**: 20  

Training Loss decreased from **1.08 → 0.17** across 20 epochs.  

- CER (Character Error Rate) and WER (Word Error Rate) were tracked across epochs.  
- Loss curves and error metrics showed exponential improvement during training.  

---

## 🚀 Results  

### On Training Speakers (S1–S20)  
- CER ≈ **0.05**  
- WER ≈ **0.16**  

### On Unseen Speakers (S21–S25)  
- Before Fine-Tuning: CER ≈ **0.21**, WER ≈ **0.44**  
- After Fine-Tuning: CER ≈ **0.05**, WER ≈ **0.13**  

📌 Fine-tuning with transfer learning drastically reduced errors and improved generalization.  

---

## 📈 Visualizations  

- 📉 Training Loss Curve  
- 📊 CER / WER Distributions  
- 🔥 Confusion Matrix (character-level)  
- 🎯 Streamlit UI for video-level predictions  

*(Add your graphs/screenshots here)*  

---

## 🖥️ Streamlit App  

An interactive Streamlit dashboard was built to:  

- Select speaker + video → Get prediction  
- Compare **Predicted vs Actual transcript**  
- Display **CER/WER metrics**  
- Evaluate dataset-wide performance with graphs (histograms, averages, confusion matrix)
## 📌 Key Learnings

Lip reading is highly challenging due to visually similar lip movements.

Fine-tuning bridged the gap between seen vs unseen speakers.

Visualization of CER/WER distributions and confusion matrices was crucial to understand model behavior.

Built a full ML pipeline: raw data → preprocessing → training → fine-tuning → evaluation → deployment.

## 🔮 Future Work

Transformer-based architectures for lip reading

Multilingual lip reading (beyond English)

Audio-visual fusion (combine audio + video for robust recognition)
## 🏗️ Project Structure
```
├── data/                  # Raw GRID dataset (videos + audios)
├── frames/                # Extracted frames (S1–S20)
├── frames_unseen/         # Extracted frames (S21–S25)
├── labels.json            # Transcripts for training speakers
├── labels_new.json        # Transcripts for unseen speakers
├── src/
│   ├── train.py          # Model training
|   ├── preprocess.py          # extract frames from video S1 to S20
|   ├── preprocess_new.py          # extract frames from video S21 to S25
|   ├── labels_extract.py    # transcript extract from S1 to S20 video
|   ├── labels_extract_new.py          # transcript extract from s21 to s25 video
│   ├── finetune.py        # Fine-tuning unseen data
│   ├── predict.py         # Predictions on training data
│   ├── predict_new.py     # Predictions on unseen data
│   ├── predict_finetuned.py # Predictions using fine-tuned model
|   ├── evaluate_results.py          # plot graph for model_performance
|   ├── evaluation_graphs.py          # evaluate graphs for CER, WER on data
|   ├── shape_predictor_68_face_landmarks.dat # used to locate the mouth region for lip reading preprocessing.
│   ├── app.py             # Streamlit UI
├── lipreading_model_final.pth          # Model saved after training
├── requirements.txt          # project related dependencies
├── lipreading_model_finetuned.pth          # Model saved after finetuning
├── results.csv            # Predictions on training data
├── results_new.csv        # Predictions on unseen speakers
├── results_finetuned.csv  # Predictions after fine-tuning
├── model_performance_summary.csv  # csv results of model in predictions
└── README.md
```
Run locally:  
```bash
streamlit run src/app.py
```






