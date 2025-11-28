# Build Environment for ManiFlow Policy from scratch

1.Prerequisites

    git clone https://github.com/geyan21/ManiFlow_Policy.git
    cd ManiFlow_Policy
    cd scripts

---

2.Install Vulkan

    sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools

---

3.Create a conda env

You can use a conda environment YAML file to create the env:

    conda env create -f conda_environment.yaml
    conda activate maniflow

Or create a conda env manually:

    conda create -n maniflow python=3.10
    conda activate maniflow

    # Install additional dependencies
    pip install -r requirements.txt

---

4.Install PyTorch3D, check the [official install instruction](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) if encountering any errors:

    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

---

5.Install flash attention (optional)

normally it is not necessary, but if you want to use it, please check the [official install instruction](https://github.com/Dao-AILab/flash-attention) for more details or run the following command:

    MAX_JOBS=4 python -m pip -v install flash-attn --no-build-isolation

---

6.Install ManiFlow as a package

    cd ManiFlow && pip install -e . && cd ..

---

7.install mujoco in `~/.mujoco`

    cd ~/.mujoco
    wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate

    tar -xvzf mujoco210.tar.gz

and put the following into your bash script (usually in `YOUR_HOME_PATH/.bashrc`). Remember to `source ~/.bashrc` to make it work and then open a new terminal.

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
    export MUJOCO_GL=egl


and then install mujoco-py (in the folder of `third_party`):

    cd YOUR_PATH_TO_THIRD_PARTY
    cd mujoco-py-2.1.2.14
    pip install -e .
    cd ../..


----

8.install sim env

    cd third_party
    <!-- cd dexart-release && pip install -e . && cd .. --> # uncomment this line if you want to use dexart, see below for more details
    cd gym-0.21.0 && pip install -e . && cd ..
    cd Metaworld && pip install -e . && cd ..
    cd rrl-dependencies && pip install -e mj_envs/. && pip install -e mjrl/. && cd ../
    cd r3m && pip install -e . && cd ../..

- download assets from [Google Drive](https://drive.google.com/file/d/1DxRfB4087PeM3Aejd6cR-RQVgOKdNrL4/view?usp=sharing), unzip it, and put it in `third_party/dexart-release/assets`. 

- download Adroit RL experts from [OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS), unzip it, and put the `ckpts` folder under `$YOUR_REPO_PATH/third_party/VRL3/`.

**⚠️ Important Note:** dexart sim env requires sapien to be `2.2.1`, which is in conflict with robotwin that requires sapien `3.0.0b1`. If you want to use both dexart and robotwin, you can create two separate conda envs with different sapien versions or manually downgrade and upgrade sapien when switching between dexart and robotwin. Otherwise, the sapien version is set to `3.0.0b1` in the `requirements.txt` for robotwin as the default.

---

9.install robotwin env

    cd third_party/RoboTwin1.0

**Download required models and assets:**
- download robotwin models from [Google Drive](https://drive.google.com/file/d/1VOvXZMWQU8-Y1-T2Si5SQLxdH6Eh8nVm/view?usp=sharing), unzip it, and put it in `third_party/robotwin1.0/`. 

- download robotwin assets from [Google Drive](https://drive.google.com/file/d/1VPyzWJYNxQUMf3KSObCyjhawIZMPZExM/view?usp=sharing), unzip it, and put it in `third_party/robotwin1.0/`. 

then copy the robotwin models and assets to `ManiFlow_Policy/ManiFlow/maniflow/env/robotwin`

---

10.Install and ⚠️ Modify mplib Library Code (only for robotwin, skip this step if you do not use robotwin)

    pip install mplib==0.1.1 sapien==3.0.0b1

Use the following command to find the path of `mplib` package and then modify the code in `planner.py` accordingly:

    MPLIB_LOCATION=$(pip show mplib | grep 'Location' | awk '{print $2}')/mplib
    PLANNER=$MPLIB_LOCATION/planner.py 

    # Comment out convex=True parameter (safe - only one instance)
    sed -i -E 's/^(\s*)(.*convex=True.*)/\1# \2/' $PLANNER

    # Remove 'or collide' from the condition  
    sed -i -E 's/(if np\.linalg\.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' $PLANNER

or you can manually modify the following line in the `planner.py` file:

    # mplib.planner (mplib/planner.py) line 71
    # remove `convex=True`

    self.robot = ArticulatedModel(
                urdf,
                srdf,
                [0, 0, -9.81],
                user_link_names,
                user_joint_names,
                # convex=True, # comment this line or remove it
                verbose=False,
            )

    # mplib.planner (mplib/planner.py) line 848
    # remove `or collide`

    if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
                    return {"status": "screw plan failed"}
    =>
    if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
                    return {"status": "screw plan failed"}

---

11.install visualizer for pointclouds (optional)

    pip install kaleido plotly
    cd visualizer && pip install -e . && cd ..

---
*This document incorporates installation procedures from [3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/INSTALL.md), [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/blob/RoboTwin-1.0/INSTALLATION.md), and other open-source projects.*