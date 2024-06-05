![Teaser](teaser.png)

# Neural Inpainting of Folded Fabrics with Interactive Editing

This repository contains a research prototype implementation of "Neural Inpainting of Folded Fabrics with Interactive Editing", Computer & Graphics.

The method proposed aims to fill holes in incomplete meshes representing fabrics. The provided files enable the user to either retrain the network or directly apply it to data using the provided weights. Additionally, the user can utilize several editing tools to guide the output predicted by the network. The repository also contains the ScarfFolds dataset used for the training.


## Requirements

You can easily create a conda environnement with all the necessary dependencies:
```bash
conda env create -f environment.yaml
conda activate NIF
```

To download

## Training

The `train.py` file contains a `conf` dictionnary representing all the available settings. The first 3 are self-explanatory.

* `lr_factor`, `lr_patience`, `lr_min` and `lr_threshold` are parameters for the optimizer scheduler
* `alpha` : weight of the data term in the loss
* `beta` : weight of the auto-intersection term in the loss
* `gamma` : weight of the gradient term in the loss

To train the network, simply type:
```bash
python train.py
```

The output folder will contain the latest weights file and the weights that minimize the loss as well as some outputs in the 

## Inference

To use the model on the data provided, use the followning command:
```bash
python inferencePolyscope.py [path/to/weights] [path/to/file] [grab | twist | pinch]
```

To test the proposed example:
```bash
python inferencePolyscope.py ./weights/standard.tar ./examples/sim1_hole.ply grab
```
