# Database3 server

This server is based on Arch Linux and equiped with mongodb and anaconda.
Other than the base installtion and the above mentioned packages, 
it also installs and configure my vim and zsh dotfiles.

## Variables

| Var             | Type    | Description                                                                                               |
|-----------------|---------|-----------------------------------------------------------------------------------------------------------|
| name            | string  | Hostname of the box                                                                                       |
| ip              | string  | IP address of the host-only adapter                                                                       |
| netmask         | string  | Subnet mask of the host-only adapter                                                                      |
| box             | string  | Name of the box, downloaded if not present on host system                                                 |
| box_url         | string  | Url where to download the box from                                                                        |
| gui             | boolean | Wether to run the VM in detached mode or not                                                              |
| memory          | int     | RAM of the VM                                                                                             |
| cpus            | int     | CPU of the VM                                                                                             |
| disksize        | string  | Disksize of the primary disk **NOTE:** requires plugin _vagrant-disksize_                                 |
| forwarded_ports | []      | Dictionary of ports to forward between the host and the VM<br>- guest: _guest port_<br> host: _host port_ |
| synced_folders  | []      | Dictionary of folders to sync between the host and the VM<br>- src: '/foo/bar'<br> dest: '/foo/bar'       |
| packages        | []      | Dictionary of packages to install on the VM                                                               |

## Requirements

```Bash
vagrant plugin install vagrant-disksize
```

## Usage

```Bash 
vagrant up
```

After launching the box you can connect to the mongodb instance by setting up
a connection for **localhost:27017**.

**TODO:** exec jupyter on startup, temp solution is to provision on each boot

Jupyter is configured to launch in the synced_folders dir.
After provisioning you will see a link. You can copy pasta the link directing to localhost into 
your webbrowser to open the Jupyter workspace.

e.g.  http://127.0.0.1:8888/?token=69d1b9...

**NOTE:** just going to 127.0.0.1:8888 might work too.

If you didn't use the link with the token and it prompts you to login, you can retrieve the token
by ssh-ing into the server and executing

```Bash
cat /home/vagrant/conda.log
```
