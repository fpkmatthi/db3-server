# -*- mode: ruby -*-
# vi: ft=ruby :

require 'rbconfig'
require 'yaml'

# Set your default base box here
DEFAULT_BASE_BOX = 'generic/arch'

#
# No changes needed below this point
#

VAGRANTFILE_API_VERSION = '2'
PROJECT_NAME = '/' + File.basename(Dir.getwd)

hosts = YAML.load_file('vagrant-hosts.yml')

# {{{ Helper functions

def is_windows
  RbConfig::CONFIG['host_os'] =~ /mswin|mingw|cygwin/
end

# Set options for the network interface configuration. All values are
# optional, and can include:
# - ip (default = DHCP)
# - netmask (default value = 255.255.255.0
# - mac
# - auto_config (if false, Vagrant will not configure this network interface
# - intnet (if true, an internal network adapter will be created instead of a
#   host-only adapter)
def network_options(host)
  options = {}

  if host.has_key?('ip')
    options[:ip] = host['ip']
    options[:netmask] = host['netmask'] ||= '255.255.255.0'
  else
    options[:type] = 'dhcp'
  end

  if host.has_key?('mac')
    options[:mac] = host['mac'].gsub(/[-:]/, '')
  end
  if host.has_key?('auto_config')
    options[:auto_config] = host['auto_config']
  end
  if host.has_key?('intnet') && host['intnet']
    options[:virtualbox__intnet] = true
  end

  options
end

def custom_synced_folders(vm, host)
  if host.has_key?('synced_folders')
    folders = host['synced_folders']

    folders.each do |folder|
      vm.synced_folder folder['src'], folder['dest'], folder['options']
    end
  end
end

# Adds forwarded ports to your Vagrant machine
#
# example:
#  forwarded_ports:
#    - guest: 88
#      host: 8080
def forwarded_ports(vm, host)
  if host.has_key?('forwarded_ports')
    ports = host['forwarded_ports']

    ports.each do |port|
      vm.network "forwarded_port", guest: port['guest'], host: port['host']
    end
  end
end

# }}}

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
  config.ssh.insert_key = false # automatically insert a keypair to use for SSH, replacing Vagrant's default insecure key if detected
  # config.ssh.keys_only = false # Only Vagrant-provided SSH  keys (no keys stored in ssh-agent).
  hosts.each do |host|
    config.vm.define host['name'] do |node|
      node.vm.box = host['box'] ||= DEFAULT_BASE_BOX
      if(host.key? 'box_url')
        node.vm.box_url = host['box_url']
      end

      node.vm.hostname = host['name']
      node.vm.network :private_network, network_options(host)
      custom_synced_folders(node.vm, host)
      forwarded_ports(node.vm, host)
      # config.vm.network "forwarded_port", guest: 3000, host: 3000
      # config.vm.network "forwarded_port", guest: 27017, host: 27017

      node.vm.provider :virtualbox do |vb|
        vb.gui = host['gui']
        vb.cpus = host['cpus'] if host.key? 'cpus'
        vb.memory = host['memory'] if host.key? 'memory'
        
        # WARNING: if the name of the current directory is the same as the
        # host name, this will fail.
        vb.customize ['modifyvm', :id, '--groups', PROJECT_NAME]
      end

      config.disksize.size = host['disksize']

      # Append the packages to install to one string
      packages=''
      host['pacman'].each do |package|
        packages.concat(package)
      end

      # Run configuration scripts for the VM
      node.vm.provision 'shell',
        path: 'provisioning/install_packages.sh',
        args: [
          host['name'],
          host['username'],
          packages
        ]

      # Append the packages to install to one string
      packages=''
      host['aur'].each do |package|
        packages.concat(package)
      end

      node.vm.provision 'shell',
        path: 'provisioning/' + host['name'] + '.sh',
        args: [
          host['name'],
          host['username'],
          packages
        ]
    end
  end
end

