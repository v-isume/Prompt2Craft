# Prompt2Craft
Code for the paper **Prompt2Craft: Generating Functional Craft Assemblies With LLMs**, published in IEEE Robotics and Automation Letters (RA-L) and presented at IROS 2026.

Vitor Hideyo Isume, Takuya Kiyokawa, Natsuki Yamanobe, Yukiyasu Domae, Weiwei Wan, Kensuke Harada

[Project page](https://v-isume.github.io/Prompt2Craft/) | [Paper](https://ieeexplore.ieee.org/document/11513915) | [arXiv](https://arxiv.org/abs/2512.04568)

## Environment Setup
The main packages are in *requirements.txt*. It was tested using Python 3.12.

For the Physics Simulation Engine, in the paper we used Nvidia Isaac SIM 4.2.0, however, in this release, we updated it to Isaac SIM 6.0.1.
Please install by following the instructions from their official repo: https://github.com/isaac-sim/IsaacSim

## Dataset Preparation
The following synsets from ImageNet-21k (Winter 2021) dataset were used:

```
bookshelf: {n02871439, n04190052}
bus: {n02924116}
chair: {n03001627}
hammer: {n03481172}
scooter: {n04149374}
skateboard: {n04225987}
table: {n03063968}
truck: {n03360300, n03690473, n04467665, n04490091}
```

**Differences with paper:** Around ~36 *truck* images are different from the original dataset. As I cannot directly distribute the ImageNet images, and because I lost their original names and source (which release/year they were from), I replaced them for now so that the dataset can be recreated when using *./scripts/rebuild_dataset.py*.

After downloading the synsets, place them all in the same folder. Then, you can reconstruct our filtered dataset by running the following commands:

```
cd Prompt2Craft
python ./scripts/rebuild_dataset.py rebuild --source "path/to/synsets/root/folder"
```

***Note:*** The script should also work if you download fewer synsets as well, you can first try it with a single one.

The additional images for the *scooter* class are in *dataset/scooter_pexels.zip*, extract those files in the same folder as the scooter images after running the script.
All the additional images were collected from Pexels website, and resized.

## Prompt

For prompting, we used batch prompting via the OpenAI API.

1. Set your OPENAI_API_KEY in your environment:
```
export OPENAI_API_KEY = "Your_API_key" # or setup in your ~./bashrc
```

2. Configure the parameters for the prompt at *config/batch_prompt_config.json*

3. Generate the prompts batch for a specific object class:
```
python ./scripts/batch_prompt.py
```

4. Configure the parameters for the batch at *config/batch_job_config.json* or pass the arguments down when executing the script.

5. Upload the batch and wait for it to finish
```
python ./scripts/upload_batch.py
```

5.a If you want to keep track of the status in the same terminal window, pass down the --wait flag:
```
python ./scripts/upload_batch.py --wait
```

 ## Mesh Generation

 Once you get the answers from the VLM, run the following script to generate the meshes:

```
python scripts/batch_mesh_gen_streamlined.py --input-jsonl /path/to/answers/jsonl --output-root /path/to/save/folder --error-root /folder/to/save/error/file --rescale-root /folder/to/save/rescaled/meshes
```

Alternatively, you can specify the parameters in *./config/mesh_gen_config.json*.
We perform a re-scaling step here to ensure that the meshes are in meters for the Physics Simulation.

Then, verify the collisions by running:

```
python scripts/check_collisions.py --in-dir /name/of/meshes/folder/
```

## Physics Simulation

To run the physics simulation, run the following:

```
python scripts/check_physics.py --in-dir /name/of/meshes/folder/ --out-dir /name/of/folder/to/save/output/
```

You can add the --headless flag if you don't need to visualize the simulation.

## Citation

```bibtex
@article{isume2026prompt2craft,
title={Prompt2Craft: Generating Functional Craft Assemblies with LLMs},
author={Isume, Vitor Hideyo and Kiyokawa, Takuya and Yamanobe, Natsuki and Domae, Yukiyasu and Wan, Weiwei and Harada, Kensuke},
journal={IEEE Robotics and Automation Letters},
year={2026},
publisher={IEEE}
}
```
