# Path-Level Network Transformation for Efficient Architecture Search

Code for the paper [Path-Level Network Transformation for Efficient Architecture Search](TODO) in ICML 2018. 

## Reference
```bash
```

## Related Projects
- [Efficient Architecture Search by Network Transformation](https://arxiv.org/abs/1707.04873) in AAAI 2018, [Code](https://github.com/han-cai/EAS).

## Dependencies

* Python 3.6 
* Pytorch 0.3.1

## Results

### CIFAR-10

|           Model          | Params | Test error (%) | 
| ----------------------- | ------------- | ----- |
| [TreeCell-A with DenseNet (N=16, k=48, G=2)](TODO) |  13.1M | 3.35 |
| [TreeCell-A with PyramidNet (N=18, alpha=84, G=2)](TODO) | 5.7M | 3.14 |
| [TreeCell-A with PyramidNet (N=18, alpha=84, G=2) + DropPath (600 epochs)](TODO) | 5.7M | 2.99 |
| [TreeCell-A with PyramidNet (N=18, alpha=84, G=2) + DropPath + Cutout (600 epochs)](TODO) | 5.7M | 2.49 |
| [TreeCell-A with PyramidNet (N=18, alpha=150, G=2) + DropPath + Cutout (600 epochs)](TODO) | 14.3M | 2.30 |

For checking these networks, please download the corresponding model files and run the following command under the folder of **code/CIFAR**:
```bash
$ python3 run_exp.py --path <nets path>
```

For example, by running
```bash
$ python3 run_exp.py --path ../../Nets/CIFAR10#PyramidTreeCellA#N=18_alpha=150#600#cutout
```
you will get
```bash
test_loss: 0.092100	 test_acc: 97.700000
```


### ImageNet

|           Model          | Multi-Add | Top-1 error (%) | 
| ----------------------- | ------------- | ----- |
| [TreeCell-B with CondenseNet (G1=4, G3=8)](TODO)| 594M | 25.4 | 

Please refer to the file **code/ImageNet/scripts.sh**.


## Architecture Search
For setting up your environment to run architecture search experiments, please refer to my previous repository, [EAS](https://github.com/han-cai/EAS/tree/master/code).

