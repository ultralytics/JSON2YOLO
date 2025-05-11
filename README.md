<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# 🚀 Introduction

Welcome to the [JSON2YOLO](https://github.com/ultralytics/JSON2YOLO) repository! This toolkit is designed to help you convert datasets in [JSON](https://www.ultralytics.com/glossary/json) format, particularly those following the [COCO (Common Objects in Context)](https://cocodataset.org/#home) standards, into the [YOLO format](https://docs.ultralytics.com/datasets/#yolo-format). The YOLO format is widely recognized for its efficiency in [real-time](https://www.ultralytics.com/glossary/real-time-inference) [object detection](https://docs.ultralytics.com/tasks/detect/) tasks.

This conversion process is essential for [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) practitioners looking to train object detection models using frameworks compatible with the YOLO format, such as [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/). Our code is flexible and designed to run across various platforms including Linux, macOS, and Windows.

[![Ultralytics Actions](https://github.com/ultralytics/JSON2YOLO/actions/workflows/format.yml/badge.svg)](https://github.com/ultralytics/JSON2YOLO/actions/workflows/format.yml)
[![Ultralytics Discord](https://img.shields.io/discord/1089800235347353640?logo=discord&logoColor=white&label=Discord&color=blue)](https://discord.com/invite/ultralytics)
[![Ultralytics Forums](https://img.shields.io/discourse/users?server=https%3A%2F%2Fcommunity.ultralytics.com&logo=discourse&label=Forums&color=blue)](https://community.ultralytics.com/)
[![Ultralytics Reddit](https://img.shields.io/reddit/subreddit-subscribers/ultralytics?style=flat&logo=reddit&logoColor=white&label=Reddit&color=blue)](https://reddit.com/r/ultralytics)

## ⚙️ Requirements

To get started with JSON2YOLO, you'll need a [Python](https://www.python.org/) environment running version 3.8 or later. Additionally, you'll need to install all the necessary dependencies listed in the `requirements.txt` file. You can install these dependencies using the following [pip](https://pip.pypa.io/en/stable/) command in your terminal:

```bash
pip install -r requirements.txt # Installs all the required packages
```

## 💡 Usage

To convert your COCO JSON dataset to YOLO format, run the `convert.py` script from your terminal. You need to specify the path to the directory containing your COCO JSON annotation files and the directory where you want to save the resulting YOLO label files.

```bash
# Example usage: Convert COCO annotations to YOLO format
python convert.py --json_dir path/to/coco/annotations --save_dir path/to/yolo/labels
```

- `--json_dir`: Path to the directory containing COCO JSON annotation files (e.g., `instances_train2017.json`).
- `--save_dir`: Path to the directory where the converted YOLO label files (`.txt`) will be saved.

This script will process the JSON files, extract bounding box information, and convert it into the YOLO format, saving one `.txt` file per image in the specified save directory. For more details on [dataset formats](https://docs.ultralytics.com/datasets/), refer to our documentation.

## 📚 Citation

If you find our tool useful for your research or development, please consider citing it:

[![DOI](https://zenodo.org/badge/186122711.svg)](https://zenodo.org/badge/latestdoi/186122711)

## 🤝 Contribute

We welcome contributions from the community! Whether you're fixing bugs, adding new features, or improving documentation, your input is invaluable. Take a look at our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) to get started. Also, we'd love to hear about your experience with Ultralytics products. Please consider filling out our [Survey](https://www.ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey). A huge 🙏 and thank you to all of our contributors!

[![Ultralytics open-source contributors](https://raw.githubusercontent.com/ultralytics/assets/main/im/image-contributors.png)](https://github.com/ultralytics/ultralytics/graphs/contributors)

## ©️ License

Ultralytics offers two licensing options to accommodate diverse needs:

- **AGPL-3.0 License**: Ideal for students and enthusiasts, this [OSI-approved](https://opensource.org/license/agpl-v3) open-source license promotes collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for details.
- **Enterprise License**: Designed for commercial use, this license permits seamless integration of Ultralytics software and AI models into commercial products and services, bypassing the open-source requirements of AGPL-3.0. For commercial inquiries, please contact us through [Ultralytics Licensing](https://www.ultralytics.com/license).

## 📬 Contact Us

For bug reports, feature requests, and contributions, please visit [GitHub Issues](https://github.com/ultralytics/JSON2YOLO/issues). For broader questions and discussions about this project and other Ultralytics initiatives, join our vibrant community on [Discord](https://discord.com/invite/ultralytics)!

<br>
<div align="center">
  <a href="https://github.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="Ultralytics GitHub"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.linkedin.com/company/ultralytics/"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="3%" alt="Ultralytics LinkedIn"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://twitter.com/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="3%" alt="Ultralytics Twitter"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://youtube.com/ultralytics?sub_confirmation=1"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="3%" alt="Ultralytics YouTube"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://www.tiktok.com/@ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="3%" alt="Ultralytics TikTok"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://ultralytics.com/bilibili"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-bilibili.png" width="3%" alt="Ultralytics BiliBili"></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">
  <a href="https://discord.com/invite/ultralytics"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-discord.png" width="3%" alt="Ultralytics Discord"></a>
</div>
