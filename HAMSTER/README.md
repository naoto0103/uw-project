# HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation

This repository contains the code for HAMSTER, a system for open-world robot manipulation using vision-language models.

## Dependencies

This project depends on the following external repositories and resources:

1. **VILA Repository**
   - Source: [NVlabs/VILA](https://github.com/NVlabs/VILA)
   - Required Commit: `da98f3b`
   - Usage: Used as a base for the vision-language model implementation
   - Note: This is an external dependency and should be cloned separately

2. **Model Checkpoint**
   - Source: [yili18/Hamster_dev](https://huggingface.co/yili18/Hamster_dev)
   - Usage: Contains the trained model weights
   - Note: This is downloaded automatically during setup

## Project Structure

```
.
├── server.py              # Custom server implementation
├── setup_server.sh        # Setup and launch script
├── gradio_server_example.py  # Example Gradio interface
├── ip_eth0.txt           # Stores the server IP address
└── VILA/                  # External VILA repository (not included)
    └── ...
```

## Usage

### Initial Setup

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Clone the VILA repository and checkout the specific commit:
```bash
git clone https://github.com/NVlabs/VILA.git
cd VILA
git checkout a5a380d6d09762d6f3fd0443aac6b475fba84f7e
cd ..
```

3. Set up the VILA environment:
```bash
cd VILA
./environment_setup.py vila
conda activate vila
cd ..
```

4. Install additional packages for the Gradio interface:
```bash
pip install gradio openai opencv-python matplotlib numpy
```

### Running the Server

1. Make sure you're in the VILA environment:
```bash
conda activate vila
```

2. Run the setup script to start the server:
```bash
./setup_server.sh
```
This will:
- Download the model checkpoint from Hugging Face
- Save the server IP address to `ip_eth0.txt`
- Set up the server with the correct configuration
- Start the server on port 8000

3. The server will be available at the IP address stored in `ip_eth0.txt` on port 8000

4. Use the Gradio interface by running:
```bash
python gradio_server_example.py
```
The Gradio interface will automatically use the IP address from `ip_eth0.txt` to connect to the server.

## Notes

- The VILA repository is an external dependency and should be kept separate from this repository
- Make sure to checkout the specific commit (`a5a380d6d09762d6f3fd0443aac6b475fba84f7e`) of VILA
- Always use the VILA environment (`conda activate vila`) when running the server
- Model checkpoints are downloaded from Hugging Face and not included in this repository
- Make sure you have sufficient disk space for the model checkpoint
- The server requires GPU support for optimal performance
- The server IP address is automatically detected and stored in `ip_eth0.txt`

## License

[Your License Here]

## Acknowledgments

- VILA: [NVlabs/VILA](https://github.com/NVlabs/VILA) (commit `a5a380d6d09762d6f3fd0443aac6b475fba84f7e`)
- Model weights: [yili18/Hamster_dev](https://huggingface.co/yili18/Hamster_dev) 