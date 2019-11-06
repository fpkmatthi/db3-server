#! /usr/bin/bash
#
# Utility functions that are useful in all provisioning scripts.

#------------------------------------------------------------------------------
# Variables
#------------------------------------------------------------------------------

# Color definitions
readonly reset='\e[0m'
readonly cyan='\e[0;36m'
readonly red='\e[0;31m'
readonly yellow='\e[0;33m'

# Set to 'yes' if debug messages should be printed.
readonly debug_output='yes'

#------------------------------------------------------------------------------
# Logging and debug output
#------------------------------------------------------------------------------


# Prints all arguments on the standard output stream
info() {
  printf "${yellow}>>> %s${reset}\n" "${*}"
}

# Prints all arguments on the standard error stream
debug() {
  if [ "${debug_output}" = 'yes' ]; then
    printf "${cyan}### %s${reset}\n" "${*}" 1>&2
  fi
}

# Prints all arguments on the standard error stream
error() {
  printf "${red}!!! %s${reset}\n" "${*}" 1>&2
}

#------------------------------------------------------------------------------
# Useful tests
#------------------------------------------------------------------------------

# Usage: files_differ FILE1 FILE2
#
# Tests whether the two specified files have different content
#
# Returns with exit status 0 if the files are different, a nonzero exit status
# if they are identical.
files_differ() {
  local file1="${1}"
  local file2="${2}"

  # If the second file doesn't exist, it's considered to be different
  if [ ! -f "${file2}" ]; then
    return 0
  fi

  local -r checksum1=$(md5sum "${file1}" | cut -c 1-32)
  local -r checksum2=$(md5sum "${file2}" | cut -c 1-32)

  [ "${checksum1}" != "${checksum2}" ]
}


#------------------------------------------------------------------------------
# SELinux
#------------------------------------------------------------------------------

# Usage: ensure_sebool VARIABLE
#
# Ensures that an SELinux boolean variable is turned on
ensure_sebool()  {
  local -r sebool_variable="${1}"
  local -r current_status=$(getsebool "${sebool_variable}")

  if [ "${current_status}" != "${sebool_variable} --> on" ]; then
    setsebool -P "${sebool_variable}" on
  fi
}

#------------------------------------------------------------------------------
# User management
#------------------------------------------------------------------------------

# Usage: create_user USERNAME
#
# Create the user with the specified name if it doesn’t exist
ensure_user_exists() {
  local user="${1}"
  local password="${2}"
  info "Ensure user ${user} exists"
  if ! getent passwd "${user}"; then
    useradd -m -s /bin/bash -U "${user}" -u 666 --groups wheel &&
      info " -> user added" ||
      error " -> failed to add user ${user}"
    echo "${user}:${password}" | chpasswd
  else
    info " -> already exists"
  fi
}

ensure_ssh_key () {
  info "Copying insecure ssh key"
  cp -pr "/home/vagrant/.ssh" "/home/${USERNAME}/"
  chown -R ${USERNAME}:${USERNAME} "/home/${USERNAME}"
}

ensure_sudo_permissions() {
  info "Setting up sudo permissions"
  echo "%${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> "/etc/sudoers.d/${USERNAME}"
}

ensure_dotfiles() {
  info "Copying dotfiles"

  if [ ! -d "/home/${USERNAME}/.oh-my-zsh/" ]; then
    info "Installing oh-my-zsh"
    curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh > install.sh
    ZSH="/home/${USERNAME}/.oh-my-zsh" sh install.sh --unattended
    git clone "https://github.com/romkatv/powerlevel10k.git" "/home/${USERNAME}/.oh-my-zsh/themes/powerlevel10k"
    chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/
    chsh -s /bin/zsh "${USERNAME}"
  fi


  ### TODO: diff -r files to check wether to copy
  cp -rT "${PROVISIONING_FILES}/dotfiles/" "/home/${USERNAME}/"
  chown -R ${USERNAME}:${USERNAME} "/home/${USERNAME}"

  ### TODO: test this
  # Compile vim-plugin YouCompleteMe
  # install_file="/home/${USERNAME}/.vim/plugged/YouCompleteMe/install.py"
  # if [ -f $install_file ]; then
  #   python  "${install_file}"
  # fi
}

ensure_yay() {
  info "Installing yay"
  if [ ! -f "/usr/bin/yay" ]; then 
    cd "/home/${USERNAME}"
    runuser -l ${USERNAME} -c 'git clone "https://aur.archlinux.org/yay.git" "/home/vagrant/yay"'
    runuser -l ${USERNAME} -c 'cd /home/vagrant/yay && yes | makepkg --noconfirm -si'
    rm -rf "/home/${USERNAME}/yay"
  fi
}

use_all_cores_for_compilation() {
  info "Use all cores for compilation"
  sed -i "s/-j2/-j$(nproc)/;s/^#MAKEFLAGS/MAKEFLAGS/" /etc/makepkg.conf
}

# Usage: ensure_group_exists GROUPNAME
#
# Creates the group with the specified name, if it doesn’t exist
ensure_group_exists() {
  local group="${1}"

  info "Ensure group ${group} exists"
  if ! getent group "${group}"; then
    info " -> group added"
    groupadd "${group}"
  else
    info " -> already exists"
  fi
}
