# h/t https://gist.github.com/8enmann/reinstall.sh
# Updated to use correct drivers for p2.xlarge instance

sudo service lightdm stop
sudo ~/nvidia/NVIDIA-Linux-x86_64-396.44.run --no-opengl-files
sudo ~/nvidia/cuda_9.2.148_396.37_linux.run --no-opengl-libs
# Verify installation
nvidia-smi
cat /proc/driver/nvidia/version
