# TTS-with-Diffusion-model

# Improving Text-to-Speech Synthesis with Diffusion Models: A D3PM Approach for Discrete Codecs

![Thesis Banner](images/banner.png) <!-- Replace with your banner or remove this line -->

## Abstract
This thesis explores advancements in Text-to-Speech (TTS) synthesis using diffusion models. By replacing traditional autoregressive models with a denoising diffusion probabilistic model (DDPM), we introduce a hybrid architecture that improves inference speed, scalability, and output quality. The proposed system leverages latent discrete representations through EnCodec and demonstrates robust zero-shot synthesis capabilities.

Key contributions include:
- Faster, parallelized generation.
- Enhanced speaker similarity and speech naturalness in zero-shot settings.
- Improved generalization to noisy and out-of-distribution data.


## Key Features
- **Diffusion Models in TTS:** A novel approach to generating discrete codec tokens using structured noise and denoising.
- **Neural Codec Integration:** Utilizing EnCodec for efficient latent space representation.
- **Non-Autoregressive Synthesis:** Faster inference through parallel token generation.

## Results
- **Evaluation:** Achieved higher audio similarity and speaker consistency compared to baseline autoregressive models.
- **Performance:** Reduced inference latency by over 40% and training time by 44%.
- **Generalization:** Demonstrated strong performance on out-of-distribution datasets like LibriSpeech.

| Model        | Inference Speed (tokens/sec) | Latency (sec) | Training Time (hours) |
|--------------|-------------------------------|---------------|------------------------|
| Baseline     | 120.27                       | 3.7           | 500                    |
| Proposed     | 211.90                       | 2.1           | 280                    |

## Getting Started
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/masters-thesis-diffusion-tts.git
