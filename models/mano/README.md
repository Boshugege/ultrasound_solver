# MANO model files

Place licensed MANO model files here:

```text
models/mano/MANO_RIGHT.pkl
models/mano/MANO_LEFT.pkl
```

Download source:

```text
https://mano.is.tue.mpg.de/
```

The MANO files require account login and license acceptance. Do not commit the
`.pkl` files; they are ignored by `.gitignore`.

After placing the files, run:

```powershell
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\preview_mano_pose.py `
  --mano-model-dir models\mano `
  --params-json configs\mano\zero_pose_right.json `
  --output-dir outputs\mano_pose_preview
```

Then run BEM:

```powershell
C:\Users\Admin\miniconda3\envs\srt\python.exe scripts\hand_dataset\run_mano_hand_bem_demo.py `
  --mano-scene-mesh outputs\mano_pose_preview\mano_scene_hand.vtk `
  --output-dir outputs\mano_hand_bem_demo
```
