# BirdCLEF+ 2025 - Species Identification from Audio

![BirdCLEF+ 2025 Header](./assets/header.png)

This repository contains the work of Team MSSP PVTR for the [BirdCLEF+ 2025](https://kaggle.com/competitions/birdclef-2025) Kaggle competition, focused on species identification from audio recordings of birds, amphibians, mammals, and insects from the Middle Magdalena Valley of Colombia.

## Our Approach

Our solution builds upon a modified baseline that achieved strong results in previous iterations of BirdCLEF. The core of our approach includes:

- **Audio Preprocessing**: Converting audio recordings to mel spectrograms with carefully tuned parameters
- **Model Architecture**: EfficientNetV2-S pretrained on ImageNet21k and fine-tuned on our dataset
- **Training Strategy**: Using a combination of Binary Cross-Entropy and Focal Loss (FocalLossBCE)
- **Ensemble Techniques**: Averaging predictions from models trained with different folds
- **Post-processing**: Applying temporal smoothing to improve prediction stability across consecutive audio segments

## Repository Structure

```
.
├── assets
│   └── header.png
├── download_data.sh
├── LICENSE
├── README.md
├── requirements.txt
└── src
    └── baseline.ipynb
```

After running the download script, the data folder will contain:

```
data/birdclef-2025/
├── recording_location.txt
├── sample_submission.csv
├── taxonomy.csv
├── test_soundscapes/
├── train_audio/
├── train_soundscapes/
└── train.csv
```

## Setup Instructions

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/BirdCLEF-2025.git
cd BirdCLEF-2025

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

We provide a script to automatically download the competition data from Kaggle:

```bash
# Install Kaggle CLI if you don't have it
pip install kaggle

# Run the download script
bash download_data.sh
```

**Prerequisites:**
- You need to have the Kaggle API credentials set up on your machine.
- If you don't have them configured, the script will show you how to set them up.

**Setting up Kaggle API credentials (if needed):**
1. Login to https://www.kaggle.com/
2. Go to 'Account' section
3. Scroll down to 'API' section and click 'Create New API Token'
4. This will download a kaggle.json file
5. Create the directory with: `mkdir -p ~/.kaggle`
6. Move the downloaded file: `mv /path/to/downloaded/kaggle.json ~/.kaggle/`
7. Set the correct permissions: `chmod 600 ~/.kaggle/kaggle.json`
8. Run the download script again

The script will automatically download and extract the competition data to the `data/` directory.

## Running the Code

### Technical Implementation Details

Our implementation in `src/baseline.ipynb` consists of several key components:

#### 1. Custom Loss Function

We use a combination of Binary Cross-Entropy and Focal Loss:

```python
class FocalLossBCE(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            bce_weight: float = 0.6,
            focal_weight: float = 1.4,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss
```

#### 2. Audio Processing

We convert audio to mel spectrograms using carefully tuned parameters:

```python
def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0,
        pad_mode="reflect",
        norm='slaney',
        htk=True,
        center=True,
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm
```

We extract 5-second segments from audio files and process them to create model inputs:

```python
def process_audio_segment(audio_data, cfg, is_training=True):
    total_samples = cfg.FS * cfg.WINDOW_SIZE
    
    if len(audio_data) < total_samples:
        # Pad if audio is too short
        audio_data = np.pad(audio_data, (0, total_samples - len(audio_data)), mode='constant')
    else:
        # Choose random window if in training mode and RANDOM_SEGMENT is enabled
        if is_training and getattr(cfg, "RANDOM_SEGMENT", False):
            max_start = len(audio_data) - total_samples
            start = np.random.randint(0, max_start + 1)
            audio_data = audio_data[start:start + total_samples]
        else:
            # Use first 5 seconds (deterministic)
            audio_data = audio_data[:total_samples]
    
    mel_spec = audio2melspec(audio_data, cfg)
    
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
        
    return mel_spec.astype(np.float32)
```

#### 3. Model Architecture

We use EfficientNetV2-S with custom classification head:

```python
class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        
        self.backbone = timm.create_model(
            cfg.model_name,  # "tf_efficientnetv2_s.in21k_ft_in1k"
            pretrained=False,  
            in_chans=cfg.in_channels,
            drop_rate=0.0,    
            drop_path_rate=0.0
        )
        
        backbone_out = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, num_classes)
        
    def forward(self, x):
        features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            features = features['features']
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)
        return logits
```

#### 4. Ensemble Inference

We load multiple model checkpoints and ensemble their predictions:

```python
def load_models(cfg, num_classes):
    models = []
    model_files = find_model_files(cfg)
    
    for model_path in model_files:
        checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))
        model = BirdCLEFModel(cfg, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(cfg.device)
        model.eval()
        models.append(model)
    
    return models
```

#### 5. Post-processing

We apply temporal smoothing to improve prediction consistency:

```python
# Temporal smoothing
for group in np.unique(groups):
    sub_group = sub[group == groups]
    predictions = sub_group[cols].values
    new_predictions = predictions.copy()
    for i in range(1, predictions.shape[0]-1):
        new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
    new_predictions[0] = (predictions[0] * 0.9) + (predictions[1] * 0.1)
    new_predictions[-1] = (predictions[-1] * 0.9) + (predictions[-2] * 0.1)
    sub_group[cols] = new_predictions
    sub[group == groups] = sub_group
```

### Running the Notebook

To run the `baseline.ipynb` notebook:

1. Ensure you have the competition data downloaded
2. Set the appropriate configuration parameters in the `CFG` class:
   - Audio parameters: `N_FFT=2048`, `HOP_LENGTH=512`, `N_MELS=256`
   - Model parameters: `model_name="tf_efficientnetv2_s.in21k_ft_in1k"`
   - Inference parameters: `batch_size=16`, `use_tta=False`
3. Run the notebook cells sequentially
4. The final prediction CSV will be saved as `submission.csv`

## Results

Our current best submissions on Kaggle:
- Version 8: Score: 0.831
- Version 7: Score: 0.830 (current code)

## Team Members

- Taha Ababou - [GitHub](https://github.com/tahababou12) | [LinkedIn](https://www.linkedin.com/in/tahaababou)
- Vajinder Kaur - [GitHub](https://github.com/VajinderKaur) | [LinkedIn](https://www.linkedin.com/in/vajinder-kaur-7ba9451b4/)
- Paul Moon - [GitHub](#) | [LinkedIn](#)
- Reese Mullen - [GitHub](https://github.com/mullenrd) | [LinkedIn](https://www.linkedin.com/in/reese-mullen-7555b2166/)
