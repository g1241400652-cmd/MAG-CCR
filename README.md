# MAG-CCR

Core implementation of **Toward Effective Multimodal Graph Pre-training via Cross-Modal Residual Reconstruction**.

MAG-CCR uses two-fold node-local prediction to determine what each modality can recover from the other. The graph encoder reconstructs the residual after both modalities at selected nodes are masked. Downstream readouts combine the original features with embeddings computed from unmasked inputs.

This release includes the residual objective, its matched full-feature control, and the node classification (NC) and link prediction (LP) readouts. The run settings cover seeds 0–4 for NC on Grocery, Toys, Movies, and RedditS and LP on Grocery and Toys. The matched NC comparison in the paper uses Grocery and RedditS.

## Quick start

Use Python 3.11 on Linux or Windows. Install the dependencies in a virtual environment, then run either task on a small synthetic graph.

~~~shell
python -m pip install -r requirements.txt
python -m mag_ccr --demo --task nc
python -m mag_ccr --demo --task lp
~~~

The demo uses a reduced training budget. The JSON output gives validation and test metrics as fractions.

## OpenMAG experiments

Use the [OpenMAG data release](https://huggingface.co/datasets/Uniguri810/OpenMAG/tree/643f9ca6e27e9be0b5d691d08468a4af2cc60a1d) at the revision linked above. Put the graph and the two 768-dimensional CLIP ViT-L/14 arrays for each dataset under the dataset's directory. For Grocery the paths are as follows.

~~~text
openmag/Grocery/GroceryGraph.pt
openmag/Grocery/text_features/Grocery_text_clip_vit_large_patch14_768d.npy
openmag/Grocery/image_features/Grocery_image_clip_vit_large_patch14_768d.npy
~~~

Run the commands below from the repository root. Use the same seed and change the target to compare the two reconstruction objectives under matched settings.

~~~shell
python -m mag_ccr --data-dir openmag --dataset Grocery --task nc --seed 0
python -m mag_ccr --data-dir openmag --dataset Grocery --task lp --seed 0
python -m mag_ccr --data-dir openmag --dataset Grocery --task lp --seed 0 --target full-feature
~~~

The default uses CPU. Full training, particularly LP validation, is substantially faster with a CUDA-enabled installation of both [PyTorch](https://pytorch.org/get-started/locally/) and [DGL](https://www.dgl.ai/pages/start.html) and the option **--device cuda**.

The code rebuilds the node and undirected relation splits from the published graphs and the seeds in the repository. LP propagation uses training relations. Evaluation uses fixed candidates with 1,000 distinct negatives per endpoint. The readouts select a checkpoint by validation accuracy (NC) or validation MRR (LP) before evaluating the test split. NC uses per-target, per-modality RMS scales. LP uses unit scales.

If you use these datasets, cite [OpenMAG](https://proceedings.mlr.press/v306/wan26h.html) and follow the terms of the upstream data sources.
