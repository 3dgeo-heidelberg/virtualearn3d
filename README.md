# virtualearn3d
Repository for the framework developed in the VirtuaLearn3D project

## Usage

### Data mining

Compute features describing the point cloud. Most machine learning models do
not extract their own features. Thus, they must be computed often before
training or applying a model.

```bash
python vl3d.py --mine data_mining_spec.json
```

**IMPLEMENTING ...**



### Model training

Train a machine learning model. The command below will often be used to start
the loop.


```bash
python vl3d.py --train model_spec.json
```
**IMPLEMENTING**




### Continue model training

Train a machine learning model from a previously trained model. The command
below will often be used during the loop.

```bash
python vl3d.py --train model_spec.json --pretrained model_file.model
```
**NOT IMPLEMENTED**




### Predictions

Use a previously trained machine learning model to compute prediction
tasks, e.g., classification or regression.

```bash
python vl3d.py --predict model_spec.json
```
**NOT IMPLEMENTED**




### Model and data evaluation

Evaluate a given model or dataset. Evaluations often imply text reports and
figures. Besides, they can also output point clouds, e.g., when evaluating the
point-wise usefulness for further training iterations.

```bash
python3 vl3d.py --eval eval_spec.json
```
**NOT IMPLEMENTED**




### Full pipeline

Run an entire pipeline. A full pipeline can support all the previous
computations. It can start computing some geometric features for a point-wise
characterization, then initialize a model, evaluate its output and the dataset,
and request further training iterations on demand until an acceptance threshold
or the maximum number of iterations has been reached.

```bash
python vl3d.py --pipeline pipeline_spec.json
```
**IMPLEMENTING ...**
