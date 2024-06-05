![Teaser](teaser.png)

# Neural Inpainting of Folded Fabrics with Interactive Editing

This repository contains a research prototype implementation of "Neural Inpainting of Folded Fabrics with Interactive Editing", Computer & Graphics.

The method proposed aims to fill holes in incomplete meshes representing fabrics. The provided files enable the user to either retrain the network or directly apply it to data using the provided weights. Additionally, the user can utilize several editing tools to guide the output predicted by the network. The repository also contains the ScarfFolds dataset used for the training.


## Requirements

Download the repo using :
```bash
git clone https://github.com/g-gisbert/Neural-Inpainting-Of-Folded-Fabric-Meshes.git
```

You can easily create a conda environnement with all the necessary dependencies:
```bash
conda env create -f environment.yaml
conda activate NIF
```

To download the weights, use the command:
```bash
python trainUnet.py
```

We recommend to use the 'standard.tar' weights for most cases. 'gravity.tar' has been trained without rotations so the prediction takes the orientation into account.
'interpen.tar' has been trained with a bigger weight on the auto-intersection loss term and should predict less auto-intersected surfaces.

## Training

The `train.py` file contains a `conf` dictionnary representing all the available settings. The first 3 are self-explanatory.

* `lr_factor`, `lr_patience`, `lr_min` and `lr_threshold` are parameters for the optimizer scheduler
* `alpha` : weight of the data term in the loss
* `beta` : weight of the auto-intersection term in the loss
* `gamma` : weight of the gradient term in the loss

To train the network, simply type:
```bash
python trainUnet.py
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
