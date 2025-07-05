# VyomAI
**Transfomer models implementation from scratch using pytorch to make it more accessible for research purposes.**

**The best way to understand is learning by doing.** 

# Examples
**Each example is in one single notebook for readability and understanding**

**Each example implemented from scratch using Pytorch 2.0 **
| Task | dataset link | Pyotch 2.0 | description
|---|---|:---:|:---:|
|[**`text classification`**](https://github.com/Ajax0564/VyomAI/blob/main/Examples/vyom-ai-classification.ipynb) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|encoder model for text classification|
|[**`Fused Encoder text classification`**](https://github.com/Ajax0564/VyomAI/blob/main/Examples/vyomai-fused-kernals-2t4.ipynb) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|Combining multiple forward operations into a single step and deriving the backward propagation, we achieve a 1.3x reduction in peak memory usage and 1.6x training speed compared to native PyTorch.|
|[**`masked language modeling`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/masked_language_modeling.ipynb) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅| encoder model pretraining with mlm style| 
|[**`electra language modeling`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/electra-pretraining.ipynb) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|encoder model pretraining with electra style| 
|[**`casual language-modeling`**](https://github.com/Ajax0564/VyomAI/blob/main/Examples/vyom-ai-decoder_clm.ipynb) | [mark-twain-books](https://www.kaggle.com/datasets/msinger007/mark-twain-books) |✅|decoder model pretraining with gpt style and kv-cache for fast inference | 
|[**`casual language-modeling fused`**](https://github.com/Ajax0564/VyomAI/blob/main/Examples/vyom-ai-decode_fused.ipynb) | [mark-twain-books](https://www.kaggle.com/datasets/msinger007/mark-twain-books) |✅|fused decoder model and kv-cache for fast inference and training, we achieve a 1.3x reduction in peak memory usage and 1.4x training speed compared to native PyTorch.  | 
|[**`Supervised fine tunning (SFT) and Direct Preference Optimization (DPO)`**](https://github.com/Ajax0564/VyomAI/blob/main/Examples/vyom-ai-llm-sft-dpo-training.ipynb) | [Huggingface dataset](https://www.kaggle.com/datasets/ajax0564/sft-dpo) |✅| Training a base decoder model via SFT and DPO to follow user instructions | 
|[**`knowledge distilation`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/Knowledge_distilation.ipynb) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|initilization of a model from pretrained model|
|[**`seq2seq modeling`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/Seq2seq-absolute.ipynb) | [Flicker-30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) |✅|seq2seq model training for image caption with kv-cache|
|[**`adapters`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/adapters.ipynb) | [clinc_oos](https://www.kaggle.com/code/ajax0564/data-for-distilation) |✅|Lora and Dora for parameter efficient tunning|
|[**`vit`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/vit.ipynb) | [Scene-classification](https://www.kaggle.com/datasets/nitishabharathi/scene-classification) |✅|visual image transformer for image classification|
|[**`detr`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/detr.ipynb) | [Global-Wheat-Detection](https://www.kaggle.com/competitions/global-wheat-detection) |✅|implementation of detr DEtection TRansformer encoder decoder model for object detection|
|[**`clip`**](https://github.com/Ajax0564/VyomAI/tree/main/Examples/clip.ipynb) | [Flicker-30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) |✅|implementation of contrastive language-image pre-training|
|[**`vision language multimodel-I`**](https://github.com/Ajax0564/VyomAI/blob/main/Examples/MultiModel_basic.ipynb)|[COCO](https://www.kaggle.com/datasets/nikhil7280/coco-image-caption) |✅|A minimmal vision-language model implementation with image-text fusion  to generate image caption with RoPE and kv-cache|
|[**`vision language multimodel-II`**](https://github.com/Ajax0564/VyomAI/blob/main/Examples/vyom-ai-multimodel-advance.ipynb) |[COCO](https://www.kaggle.com/datasets/nikhil7280/coco-image-caption) |✅|A Multimodel implementation with image-text fusion  to generate caption of  image  with RoPE and kv-cache  which can we extended to visual question answering, open vocabulary object detection, optical character recognition|
|[**`vision language multimodel-II-RPAD`**](https://github.com/Ajax0564/VyomAI/blob/main/Examples/vyom-ai-accelerate-multimodel-rpad.ipynb) |[COCO](https://www.kaggle.com/datasets/nikhil7280/coco-image-caption) |✅|A Multimodel implementation with right padding image-text fusion  to generate caption of  image  with RoPE and kv-cache  which can we extended to visual question answering, open vocabulary object detection, optical character recognition|
|[**`Paligemma`**](https://github.com/Ajax0564/VyomAI/blob/main/Examples/paligemma.ipynb) | [Flicker-30k](https://www.kaggle.com/datasets/adityajn105/flickr30k) |✅|Scratch implementation of Paligemma a Multimodel from Google-AI|
****More to come**


# Usage
```python
from VyomAI import EncoderModel, EncoderForMaskedLM
from VyomAI import EncoderConfig
config = EncoderConfig()
encoder = EncoderModel(config,pos_embedding_type='rope')
#pos_embedding_type supported: Absolute, sinusoidal, RoPE
#attention_type supported: gqa, Vanila

```
## More About VyomAI

[Learn the basics of PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html)

At a granular level, it support the following components:
| Component | Description |
| ---- | --- |
| **Encoder** | Text encoder model with Bert like architecture that support absolute, sin,Rope embedding and GQA , Vanila attention|
| **Decoder** | Text decoder model with GPT like architecture that support absolute, sin,RoPE embedding and GQA , Vanila attention and KV-Cache for fast inference |
| **Seq2Seq** |Model with Bart like architecture that support absolute, sin,RoPE embedding and GQA, Vanila attention and KV-Cache for fast inference encoder can be text or image type |
| **VisionEncoder** | Model with Vit like architecture for image encoding ****more to come** |
|**Multimodel** | A Minimal vision-language model ****more to come** |



## Improvement and Contribution
We appreciate all contributions.
If you want to contribute new features, utility functions, or tutorials please  open an issue and discuss the feature with us.



## Resources
**Some helpfull learning resources**
- [1] https://www.youtube.com/@stanfordonline
- [2] https://d2l.ai/
- [3] https://pytorch.org/tutorials/

## References
- [1] https://github.com/huggingface/transformers
- [2] https://github.com/facebookresearch/detr
- [3] https://github.com/meta-llama/llama3
