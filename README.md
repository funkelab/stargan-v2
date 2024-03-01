
## StarGAN v2 - Pytorch Implementation modified for use in QuAC

## Software installation
Clone this repository:

```bash
git clone https://github.com/clovaai/stargan-v2.git
cd stargan-v2/
```

Install the dependencies:
```bash
conda create -n stargan-v2 python pytorch torchvision cudatoolkit -c pytorch
conda activate stargan-v2
pip install -e ".[dev]"
```

## Training
Here is an example of how to train a network (restarting at a defined iteration, which assumes that it has been started already).


```bash
python -u main.py --resume_iter $iteration --img_size 224 --num_domains 5 \
 --randcrop_prob 0.0 --w_hpf 0.0 --lambda_ds 0.0 --mode train \
 --train_img_dir $datadir/train --val_img_dir $datadir/val/ \
 --sample_dir $outputdir/samples --checkpoint_dir $outputdir/checkpoints \
 --eval_dir $outputdir/eval --batch_size 4 --val_batch_size 16 \
 --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
 --classifier_checkpoint /path/to/torchscript/classifier/checkpoint.pt
```

## Evaluation metrics
To evaluate StarGAN v2 using [Learned Perceptual Image Patch Similarity (LPIPS)](https://arxiv.org/abs/1801.03924) as well as on its ability to fool a pre-trained classifier, run the following command:


```bash
python -u main.py --img_size 128 --num_domains 5 \
 --randcrop_prob 0.0 --w_hpf 0.0 --mode styles --resume_iter=$1 \
 --train_img_dir $datadir/train/ --val_img_dir $datadir/val/ \
 --sample_dir $outputdir/samples --checkpoint_dir $outputdir/checkpoints \
 --eval_dir $outputdir/eval --val_batch_size 50 \
 --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 \
 --classifier_checkpoint /path/to/torchscript/classifier/checkpoint.pt
```


## License
The source code, pre-trained models, and dataset are available under [Creative Commons BY-NC 4.0](https://github.com/clovaai/stargan-v2/blob/master/LICENSE) license by NAVER Corporation. You can **use, copy, tranform and build upon** the material for **non-commercial purposes** as long as you give **appropriate credit** by citing our paper, and indicate if changes were made. 

For business inquiries, please contact clova-jobs@navercorp.com.<br/>	
For technical and other inquires, please contact yunjey.choi@navercorp.com.


## Citation
If you find this work useful for your research, please cite our paper:

```
@inproceedings{choi2020starganv2,
  title={StarGAN v2: Diverse Image Synthesis for Multiple Domains},
  author={Yunjey Choi and Youngjung Uh and Jaejun Yoo and Jung-Woo Ha},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Acknowledgements
We would like to thank the full-time and visiting Clova AI Research (now NAVER AI Lab) members for their valuable feedback and an early review: especially Seongjoon Oh, Junsuk Choe, Muhammad Ferjad Naeem, and Kyungjune Baek. We also thank Alias-Free GAN authors for their contribution to the updated AFHQ dataset.
