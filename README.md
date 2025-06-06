
# NVIDIA GPU Overclocking on Ubuntu 22.04

## 🚀 Enable NVIDIA GPU Overclocking on Ubuntu 22.04 with Coolbits

This section provides step-by-step instructions to enable NVIDIA GPU overclocking and fan control on Ubuntu 22.04 using Coolbits.

### ✅ Prerequisites

- **NVIDIA proprietary driver must be installed**  
  Check installation using:

  ```bash
  nvidia-smi
  ```

  If the command returns GPU details, you're good to go.

- **Backup your system configuration**  
  Overclocking involves system-level changes. Proceed with caution and always back up important data.

---

### 🛠️ Step-by-Step: Enabling Coolbits


1. **Edit the X configuration file**:

   ```bash
   sudo vim /etc/X11/xorg.conf
   ```

2. **Modify or insert the `Device` section as follows**:

   ```text
   Section "Device"
       Identifier "NVIDIA Card"
       Driver     "nvidia"
       Option     "Coolbits" "28"
   EndSection
   ```

3. **Edit Xwrapper config to allow GUI apps as root**:

   ```bash
   sudo nano /etc/X11/Xwrapper.config
   ```

   Replace contents with:

   ```text
   allowed_users=anybody
   needs_root_rights=yes
   ```

4. **Update file permissions**:

   ```bash
   sudo chmod 2644 /etc/X11/Xwrapper.config
   ```

5. **Reboot your system**:

   ```bash
   sudo reboot
   ```

6. **Set overclocking offset**:

   ```bash
   nvidia-settings -a "GPUGraphicsClockOffset[4]=100"
   ```
---

### 📚 References

- NeuronVM Guide: [Enable Coolbits on Ubuntu 22.04](https://neuronvm.com/docs/enable-coolbits-on-ubuntu-22-04/)
- NVIDIA Documentation: [X Config Options](https://download.nvidia.com/XFree86/Linux-x86_64/460.39/README/xconfigoptions.html)
