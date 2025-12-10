# Target-Conditioned-Shape-Based-Object-Search-Using-Tactile-Sensing-
This repository contains the project code for ECE 699: Robot Learning.  
The project explores tactile-based object search and recognition using reinforcement learning and simulation in MuJoCo.

---

## Environment Setup (For macOS (Apple Silicon))


### 1. Create a Conda Virtual Environment
```bash
conda create -n rlsim python=3.11 -y
conda activate rlsim
```

### 2. Install MuJoCo and other dependencies

Install the required system and Python dependencies:

```bash
brew install cmake glfw

pip install mujoco==3.1.6 mujoco-python-viewer gymnasium numpy pillow imageio matplotlib \
torch torchvision torchaudio 'stable-baselines3[extra]==2.3.2' pandas scikit-learn tqdm tensorboard pyyaml seaborn
```
### 3. Install the MuJoCo Desktop App (optional but recommended)

Download the .dmg installer for macOS from
🔗 https://github.com/google-deepmind/mujoco/releases

Open the .dmg and drag MuJoCo.app into your Applications folder.

### 4. Clone and Organize the Project Repository
Clone this repository and navigate into it:
```
git clone https://github.com/<your-username>/Target-Conditioned-Shape-Based-Object-Search-Using-Tactile-Sensing.git
cd Target-Conditioned-Shape-Based-Object-Search-Using-Tactile-Sensing
```

<pre> ```
project/
│
├── main_scene.py                 # Python launcher for MuJoCo scenes
├── panda_scene.xml               # Scene with Franka Panda arm + table + objects
├── shadow_hand_scene.xml         # Scene with Shadow Hand setup for tactile probing
│
├── mujoco_menagerie/             # Prebuilt robot models from DeepMind’s MuJoCo Menagerie
│   │
│   ├── franka_emika_panda/
│   │   ├── panda.xml             # Panda arm model definition
│   │   └── assets/               # Meshes and textures for the Panda robot
│   │       ├── panda_link0.obj
│   │       ├── panda_link1.obj
│   │       └── ...
│   │
│   └── shadow_hand/
│       ├── left_hand.xml         # Shadow Hand model definition
│       └── assets/               # Meshes and textures for the Shadow Hand
│           ├── forearm_0.obj
│           ├── palm.obj
│           ├── f_proximal.obj
│           └── ...
│
└── assets/                       # (Optional) additional custom assets for your own scenes
    ├── textures/
    └── meshes/
``` </pre>

### 5. Run the Final Tactile Search System (inside `final_scene/`)

All final runnable scripts for this project are located in the `final_scene/` directory.  
Navigate into the folder:

```bash
cd final_scene
```
#### 5.1 Run Baseline Manual Active Search

To run the manual baseline version, run:

```bash
mjpython main.py
```

When prompted:

```bash
Select mode (manual/ppo/train_ppo): manual
Enter target object: cube
```

This will:

* Load the MuJoCo tactile scene (Search_scene.xml)

* Touch each object (sphere → cube → cylinder → cone)

* Generate pseudo-GelSight images during contact

* Fuse classifier predictions incrementally

* Stop early if the target is confidently identified

#### 5.2 Multi-Object PPO Tactile Search (Reinforcement Learning)
This mode runs the full target-conditioned tactile search using a trained PPO policy.

Run:

```bash
mjpython multi_object_search_ppo.py
```

When prompted:

```bash
Enter target object for PPO multi-object search: cube
Enter PPO model path: 1000_ppo_policy.pth
```


