#! /usr/bin/bash
#------------------------------------------------------------------------------
# Bash settings
#------------------------------------------------------------------------------

# abort on nonzero exitstatus
set -o errexit
# abort on unbound variable
set -o nounset
# don't mask errors in piped commands
set -o pipefail

#------------------------------------------------------------------------------
# Variables
#------------------------------------------------------------------------------

# Location of provisioning scripts and files
export readonly PROVISIONING_SCRIPTS="/vagrant/provisioning"
# Location of files to be copied to this server
export readonly PROVISIONING_FILES="${PROVISIONING_SCRIPTS}/files"
export readonly HOSTNAME=$1
export readonly USERNAME=$2
export readonly AUR_PACKAGES=$3

#------------------------------------------------------------------------------
# "Imports"
#------------------------------------------------------------------------------

# Utility functions
source ${PROVISIONING_SCRIPTS}/util.sh
# Actions/settings common to all servers
source ${PROVISIONING_SCRIPTS}/common.sh

#------------------------------------------------------------------------------
# Provision server
#------------------------------------------------------------------------------

info "Starting server specific provisioning tasks for server ${HOSTNAME}"

# use_all_cores_for_compilation

# info "Updating system and installing programs"
# pacman -Syu --noconfirm
# pacman -S git vim zsh fzf base base-devel go python cmake --noconfirm --needed

# ensure_yay

if [ ! -d "/home/${USERNAME}/.oh-my-zsh/" ]; then
  info "Installing oh-my-zsh"
  curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh > install.sh
  ZSH="/home/${USERNAME}/.oh-my-zsh" sh install.sh --unattended
  git clone "https://github.com/romkatv/powerlevel10k.git" "/home/${USERNAME}/.oh-my-zsh/themes/powerlevel10k"
  chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/
  chsh -s /bin/zsh "${USERNAME}"
fi

ensure_dotfiles

# Install AUR packages
runuser -l "${USERNAME}" -c "yay -S --noconfirm --needed ${AUR_PACKAGES}"

# Enable mongodb service on startup
systemctl enable --now mongodb

# change 127.0.0.1 to 0.0.0.0
sed -i 's/127\.0\.0\.1/0.0.0.0/' /etc/mongodb.conf

# restart service
systemctl restart mongodb.service

# TODO: configure clean start/stop
### https://wiki.archlinux.org/index.php/MongoDB 

### TODO: install and configure anaconda
### Ref: https://linoxide.com/linux-how-to/install-python-anaconda-5-arch-linux-4-11-7-1/
present_in_files="/${USERNAME}/provisioning/files/install_conda.sh"
if [ ! -f "${present_in_files}" ]; then
  # download
  info "downloading anaconda"
  curl https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh > install_conda.sh
fi

# copy to home dir
cp "/vagrant/provisioning/files/install_conda.sh" "/home/${USERNAME}/install_conda.sh"

# install
### WARNING: requires manual intervention so don't run in Vagrant
### Towards the end, when it asks if you want to prepend Anaconda to your OS’s PATH variable, select ‘no’. In Arch Linux we need to Prepend Anaconda to your Path variable manually. This will make running Conda commands a lot easier.
info "installing anaconda"
sh install_conda.sh -b

# You have chosen to not have conda modify your shell scripts at all 
# To activate conda's base environment in your current shell session:
# NOTE: change ".zsh" to an extension matching your shell
eval "$("/home/${USERNAME}/anaconda3/bin/conda" shell.zsh hook)"

# To install conda's shell functions for easier access, first activate, then:
conda init

# If you'd prefer that conda's base environment not be activated on startup,
# set the auto_activate_base parameter to false:
conda config --set auto_activate_base false

# added by Anaconda3 installer
export PATH="/usr/local/anaconda/bin:$PATH"

# Test installation
conda list

# TODO: launch jupyter after boot with systemd or cron
# NOTE: to run a jupyter notebook, use "--ip=0.0.0.0" so it listens to connections coming from the host
jupyter notebook --ip=0.0.0.0 --notebook-dir=/vagrant/provisioning/files/notebooks > conda.log 2>&1 &
#cat > "/home/${USERNAME}/launch_jupyter_notebook.sh" << EOF
##!/bin/sh
#eval "$("/home/${USERNAME}/anaconda3/bin/conda" shell.zsh hook)"
#conda init
#export PATH="/usr/local/anaconda/bin:$PATH"
#jupyter notebook --ip=0.0.0.0 --notebook-dir=/vagrant/provisioning/files/notebooks > conda.log 2>&1 &
#EOF

#chmod +x "/home/${USERNAME}/launch_jupyter_notebook.sh"

#cat > /etc/systemd/system/jupyter.service << EOF
#[Unit]
#Description=Start Jupyter notebook in /vagrant/provisioning/files/notebooks

#[Service]
#ExecStart=/home/${USERNAME}/launch_jupyter_notebook.sh
#Restart=on-failure

#[Install]
#WantedBy=multi-user.target
#EOF

#systemctl enable jupyter.service
#systemctl start jupyter.service

