> [!WARNING]
> **Work in Progress**
>
> This project is under active development. It is currently **not functional** and should not be used yet.

# GeneIndex

### Automatically indexing 17M+ scans of Polish parish registers, using self-supervised pretraining to work around scarce labeled data.
The project aims to produce structured and reliable indexes (key information e.g. names and surnames) while being scalable and cost-effective.

## 🧩 Problem Statement

Even state-of-the-art multimodal LLMs fail to produce reliable, structured output because of these caveats:

- **The text itself is old and hard to read.** These registers were written a long ago, often in cursive script and in the scribe's own local hand. The data span multiple centuries and languages (Latin, German, Polish).
- **Labels aren't tied to image locations.** Labels come from an external, volunteer-built genealogical index, and each one references a record only by its book, year, and record number - not by its position in a scan. Turning a label into usable training data means first locating the matching entry in the whole book ourselves.
- **Entries have no fixed shape.** A single record can be one line or span multiple pages, so the detector must be very versatile.
- **Layout is inconsistent.** Record numbers and years - the keys needed to link an entry to its label - appear in different positions depending on the register: in the margin, mid-page, or embedded in the running text.
- **Errors cascade silently.** Matching a detected record to its label by record number is a sequential process - a single misread number can propagate incorrect labels to every subsequent entry until the sequence resets, without any obvious signal that something went wrong.
- **Most of the text is boilerplate.** The legal/religious formula language dominates each entry, while the only fields that actually matter (child, mother, father) make up a small fraction of it - and standard language-model-assisted decoding tends to "correct" rare historical surnames toward more common words, which is exactly the wrong failure mode here.

## 🔍 Project overview

![Diagram presenting overview of the project](./Project%20Overview.drawio.svg)

## ⏳ Progress

- [x] Data downloader
- [ ] Autoencoder (AE)
  - [x] Training dataset
  - [x] Architecture implementation
  - [ ] Model training
- [ ] Record splitter (RS)
  - [ ] Training dataset
  - [ ] Architecture implementation
  - [ ] Model training
- [ ] Match labels to record
  - [ ] Extract year and record numbers
  - [ ] Error correction
- [ ] HTR Model
  - [ ] Training dataset
  - [ ] Model selection
  - [ ] Model fine-tuning

## ✍️ Design decisions

The available number of labels is very large - **around 67M**, but labels cannot be easily linked with the scans:
1. Scans consist of multiple (sometimes as many as 20) entries.
2. Labels at most point to a specific register (containing a few hundred scans).

So I had to create a detector that splits a scan into single records (called ***Record Splitter*** or ***RS***), and also it must detect record number and year to allow for identification of each record.

Because of the extremely large (**roughly 17M+**) number of available scans, I decided to train a self-supervised autoencoder (called ***AE***) and then later fine-tune it as a backbone for the detector. This allows for better generalization but increases complexity and model size.

As an AE I chose **Masked Autoencoder**[1] with a few modifications inspired by **ViTDet**[2]:
- **Hybrid attention (dormant during pretraining)** - The attention module supports alternating local/global attention. This mode only activates once the encoder is repurposed as the Record Splitter's backbone. During AE pretraining itself, attention runs fully global - at this stage the sequence is short and scrambled and windowing isn't needed.
- **Small decoder** - My decoder consists of only 2 blocks of transformer and convolutional upscaler. I find that it better tolerates the very diverse strokes by smoothing it out. It also moves the heavy-lifting to the encoder that later should perform better as a backbone.

## 📊 Data & Ethical Considerations

The scans and labels used for training were taken from [The Polish Genealogical Society](https://genealodzy.pl) site which are freely available for non-commercial use.

## 📜 References

1. Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick, 
*Masked Autoencoders Are Scalable Vision Learners*, https://arxiv.org/abs/2111.06377

2. Yanghao Li, Hanzi Mao, Ross Girshick, Kaiming He,
*Exploring Plain Vision Transformer Backbones for Object Detection*, https://arxiv.org/abs/2203.16527