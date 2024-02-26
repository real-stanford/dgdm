# Dynamics-Guided Diffusion Model for Robot Manipulator Design

### [Paper](https://arxiv.org/abs/2402.15038) | [Website](https://dgdm-robot.github.io) | [Video](https://www.youtube.com/watch?v=0m5nTWgHULg)
[Xiaomeng Xu](https://xxm19.github.io/), [Huy Ha](https://www.cs.columbia.edu/~huy/), [Shuran Song](https://shurans.github.io/)

### Dependencies
Required packages can be installed by:
```
pip install -r requirements.txt
```

## Data Preparation

### Download object dataset
#### 2D objects
Download 2D object icons from [Icons50 dataset](https://www.kaggle.com/datasets/danhendrycks/icons50).

#### 3D objects
Download 3D object meshes from [MuJoCo scanned object dataset](https://github.com/kevinzakka/mujoco_scanned_objects).

### Generate simulation data
Replace ```OBJECT_DIR``` in ```sim/sim_2d.py``` and ```sim/sim_3d.py``` with the directory to object dataset.
#### 2D
```
bash sim/run_sim_2d.sh
```
#### 3D
```
bash sim/run_sim_3d.sh
```

## Training
[Download pretrained model checkpoints](https://drive.google.com/drive/folders/1jjC6G5Qv_ZkJwTjk2mCBkSyXkZu_w5EB?usp=sharing)
### Train Dynamics Model
#### 2D
```
bash dynamics/train_dynamics_2d.sh
```
#### 3D
```
bash dynamics/train_dynamics_3d.sh
```

### Train Diffusion Model
#### 2D
```
bash generator/train_diffusion_2d.sh
```
#### 3D
```
bash generator/train_diffusion_3d.sh
```

## Inference
### Generate Task-Specific Manipulators
#### 2D
```
bash generator/guided_sample_2d.sh
```
#### 3D
```
bash generator/guided_sample_3d.sh
```

## Citation
If you find DGDM useful for your work, please cite:
```
@misc{xu2024dynamicsguided,
	title={Dynamics-Guided Diffusion Model for Robot Manipulator Design}, 
	author={Xiaomeng Xu and Huy Ha and Shuran Song},
	year={2024},
	eprint={2402.15038},
	archivePrefix={arXiv},
	primaryClass={cs.RO}
}
```


## Contact
If you have any questions, please feel free to contact Xiaomeng Xu (xuxm@stanford.edu)