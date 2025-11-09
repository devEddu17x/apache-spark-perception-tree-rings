# Python 3.10 Setup Guide for Arch Linux / Manjaro

This guide explains how to install and configure Python 3.10.13 on Arch Linux and Manjaro systems. This specific version is required to match the Python version used in the Spark Docker containers.

## Why Python 3.10?

The Apache Spark Docker images use Python 3.10, so your local development environment needs the same version to ensure compatibility when running PySpark applications.

## Installation Steps

### 1. Install pyenv

`pyenv` is a Python version manager that allows you to easily install and switch between multiple Python versions.

```bash
yay -S pyenv
```

### 2. Install Python 3.10.13

Use pyenv to install the specific Python version:

```bash
pyenv install 3.10.13
```

This may take a few minutes as it compiles Python from source.

### 3. Set Python 3.10.13 as Local Version

Navigate to your project directory and set Python 3.10.13 as the local version:

```bash
pyenv local 3.10.13
```

This creates a `.python-version` file in your project directory.

### 4. Configure Shell Environment

Add pyenv initialization to your `.bashrc`:

```bash
echo -e '\n# pyenv setup\nexport PYENV_ROOT="$HOME/.pyenv"\nexport PATH="$PYENV_ROOT/bin:$PATH"\neval "$(pyenv init --path)"\neval "$(pyenv init -)"' >> ~/.bashrc
```

### 5. Restart Your Terminal

Close and reopen your terminal for the changes to take effect, or run:

```bash
source ~/.bashrc
```

### 6. Verify Installation

Check that Python 3.10.13 is active:

```bash
python --version
```

You should see:

```
Python 3.10.13
```

## Adaptation for Other Linux Distributions

### Ubuntu/Debian

```bash
# Install dependencies
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash

# Then follow steps 2-6 above
```

### Fedora

```bash
# Install dependencies
sudo dnf install -y make gcc zlib-devel bzip2 bzip2-devel \
readline-devel sqlite sqlite-devel openssl-devel tk-devel \
libffi-devel xz-devel

# Install pyenv
curl https://pyenv.run | bash

# Then follow steps 2-6 above
```

### openSUSE

```bash
# Install dependencies
sudo zypper install -y gcc automake bzip2 libbz2-devel \
xz xz-devel openssl-devel ncurses-devel readline-devel \
zlib-devel tk-devel libffi-devel sqlite3-devel

# Install pyenv
curl https://pyenv.run | bash

# Then follow steps 2-6 above
```

## Troubleshooting

### pyenv command not found after installation

Make sure you've added pyenv to your PATH and restarted your terminal. Run:

```bash
source ~/.bashrc
```

### Build fails during `pyenv install`

Install the necessary build dependencies for your distribution (see adaptation section above).

### Wrong Python version still active

Ensure you're in the project directory where `.python-version` exists, or run:

```bash
pyenv local 3.10.13
```

## Next Steps

Once Python 3.10.13 is installed and active, return to the main [README.md](README.md) to continue with the project setup.
