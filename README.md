# Prompt2Craft
Official repository for the paper "Prompt2Craft: Generating Functional Craft Assemblies With LLMs"

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

**Differences with paper:** Around ~36 *truck* images are different from the original dataset. As I cannot directly distribute the ImageNet images, and because I lost their original names and source (which release they were from), I replaced them for now so that the dataset can be recreated when using *./scripts/rebuild_dataset.py*.

After downloading the synsets, place them all in the same folder. Then, you can reconstruct our filtered dataset by running the following commands:

```
cd Prompt2Craft
python ./scripts/rebuild_dataset.py rebuild --source "path/to/synsets/root/folder"
```

***Note:*** The script should also work if you download fewer synsets as well, you can first try it with a single one.

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

