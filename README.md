# Multi-task-learning-for-road-perception

## Datasets
  - [Our dataset](https://drive.google.com/drive/folders/1S4XyyC9LbPZ8GwEH9fwWvNxKMKPA28bZ?usp=sharing)

  Train
  ```
  dataset - train - images - ....jpg
          |       | DriveArea - ....png
          |       | laneline - ....png
          |       | laneline - ....png
          |       - object - ....json
          |
          - val - images - ....jpg
                | DriveArea - ....png
                | laneline - ....png
                | laneline - ....png
                - object - ....json
  ```
  Predict
  ```
  inference - test1 - images - ....jpg
                    | DriveArea - ....png
                    | laneline - ....png
                    | laneline - ....png
                    | object - ....json
                    - oriimg - ....jpg 
             
  ```
## Weight 
  Download from [here](https://drive.google.com/file/d/1qaYYTDYY_wXv5YnRQ6MbrWRLeyv4CVVA/view?usp=sharing)

## Requirement
  This codebase has been developed with
  ```
    Python 3.9
    Cuda 12
    Pytorch 2.0.1
  ```
  See requirements.txt for additional dependencies and version requirements.
  ```shell
    pip install -r requirements.txt
  ```

## main command
  You can change the data use, Path, and Classes from [here](/data).

  ### Train
  ```shell
  python train.py
  ```
  ### Test
  ```shell
  python test.py
  ```
  ### Predict
  ```shell
  python demo.py
  ```
  ### Tensorboard
  ```shell
    tensorboard --logdir=runs
  ```

## Argument
  ### Train
  | Source           |   Argument                  |     Type    | Notes                                                                        |
  | :---             |    :----:                   |     :----:  |   ---:                                                                       |
  | hyp              | 'hyp/hyp.scratch.yolop.yaml'| str         | hyperparameter path                                                          |
  | DoOneHot         | False                       | bool        | do one hot or not                                                            |
  | useSplitModel    | False                       | bool        | use multi resnet do feature extract                                          |
  | tokensize        | 32                          | int         | size of the tokens                                                           |
  | data             | 'data/multi.yaml'           | str         | dataset yaml path                                                            |
  | logDir           | 'runs/train'                | str         | log directory                                                                |
  | epochs           | 500                         | int         | number of epochs to train for                                                |
  | val_start        | 0                           | int         | start do validation                                                          |
  | val_freq         | 1                           | int         | How many epochs do one time validation                                       |
  | batch_size       | 16                          | int         | number of images per batch                                                   |
  | workers          | 6                           | int         | maximum number of dataloader workers                                         |
  | device           | ''                          | None        | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu         |
  ### Test
  | Source           |   Argument                  |     Type    | Notes                                                                        |
  | :---             |    :----:                   |     :----:  |   ---:                                                                       |
  | hyp              | 'hyp/hyp.scratch.yolop.yaml'| str         | hyperparameter path                                                          |
  | DoOneHot         | False                       | bool        | do one hot or not                                                            |
  | useSplitModel    | False                       | bool        | use multi resnet do feature extract                                          |
  | tokensize        | 32                          | int         | size of the tokens                                                           |
  | data             | 'data/multi.yaml'           | str         | dataset yaml path                                                            |
  | weights          | './weights/epoch-200.pth'   | str         | model.pth path(s)                                                            |
  | logDir           | 'runs/train'                | str         | log directory                                                                |
  | batch_size       | 15                          | int         | 	number of images per batch                                                  |
  | workers          | 6                           | int         | maximum number of dataloader workers                                         |
  | device           | ''                          | None        | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu         |
  ### Predict
  | Source           |   Argument                  |     Type    | Notes                                                                        |
  | :---             |    :----:                   |     :----:  |   ---:                                                                       |
  | hyp              | 'hyp/hyp.scratch.yolop.yaml'| str         | hyperparameter path                                                          |
  | DoOneHot         | False                       | bool        | do one hot or not                                                            |
  | useSplitModel    | False                       | bool        | use multi resnet do feature extract                                          |
  | tokensize        | 32                          | int         | size of the tokens                                                           |
  | data             | 'data/multi.yaml'           | str         | dataset yaml path                                                            |
  | source           | './inference/val'           | str         | inference file path                                                          |
  | weights          | './weights/epoch-200.pth'   | str         | model.pth path(s)                                                            |
  | logDir           | 'runs/train'                | str         | log directory                                                                |
  | device           | ''                          | None        | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu         |
  | draw             | 'False'                     | bool        | save the Predict result                                                      |