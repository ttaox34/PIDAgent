# PID Agent

## 环境安装

### Windows

```shell
conda create -n pidagent python=3.9
conda activate pidagent
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install gym matplotlib
```

## 训练

### TD3

```shell
python rl_td3.py --ckpt "Checkpoint Name" --episodes 15000
```

支持从已有的 Checkpoint 恢复训练进度：

```shell
python rl_td3.py --resume "Checkpoint Name" --episodes 15000
```

## 测试

### TD3

会自动使用 episodes 最大的 Checkpoint 产生一个新环境下的轨迹。轨迹曲线将保存在 Checkpoint 所在的目录下。

```shell
python test_td3.py --ckpt "Checkpoint Name"
```