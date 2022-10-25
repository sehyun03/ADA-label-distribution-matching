# ADA-label-distribution-matching
Official code for ECCV 2022 paper "Combating Label Distribution Shift forActive Domain Adaptation"

#### Requirements

Install the required packages using anaconda by creating envrironment with "lamda.yml" file.
```
conda env create -f lamda.yml
```

#### Dataset preparation
Download the dataset using the following link and unzip them in `data/` folder.

Office-Home: http://hemanthdv.org/OfficeHome-Dataset/ <br />

Rename the dataset folder from "OfficeHomeDataset_10072016" into "office_home", and also rename the "Real World" within the dataset into "Real".
To run OfficeHome-RSUT, create a soft link to the "office_home" folder named as "office_home_rsut" in `data/`.

#### Evaluation
Download the trained weight files from this link (https://drive.google.com/file/d/1MiNOpYHFgr62B8X5qXexq_qVklppEBK6/view?usp=sharing) and put them in `checkpoint/` folder.
* You can evalute the result of Table 1 by running:
```
bash eval_tab1.sh
```
* You can evalute the result of Table 2 by running:
```
bash eval_tab2.sh
```
The provided domains for office_home: Art, Clipart, Product, Real
The provided domains for office_home_rsut: source (Clipart_RS, Product_RS, Real_RS), target (Clipart_UT, Product_UT, Real_UT)

#### Training
* Pretrain model on the source domain data by running:
```
python main.py --method SOURCE_ONLY --bs 32 --dataset <dataset> --source <source> --target <target>
```
  The trained weights will be saved at `checkpoint/` folder.
  You can name each experiment by using `--session` option.
  You can check the training logs from wandb.

* Train LAMDA using the pretrained source domain weights (we provide source pretrained weights in the above link).
```
python main.py --method DANN_ESTIMATED_SEMI_PMMD_ONLINE --resume <path-to-checkpoint> --dataset <dataset> --source <source> --target <target>
```

#### Other datasets
You can also train DomainNet and VisDA-2017 using the exact same commands, and the datasets can be downloaded with following links.
DomainNet: http://ai.bu.edu/M3SDA/ (cleaned version) <br />
VisDA-2017: https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification <br />
Please check `data/txt` for further configuration of each dataset.
