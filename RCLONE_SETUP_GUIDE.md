# Rclone Setup Guide for Arch Linux / Manjaro

This guide explains how to install and configure Rclone and Rclone Browser on Arch Linux and Manjaro systems to access the Cloudflare R2 bucket used for the Apache Spark project.

## Why Rclone?

Since team members don't have direct access to the Cloudflare R2 dashboard for security reasons, Rclone provides a secure way to view and manage bucket contents using provided access tokens. This allows you to browse the distributed dataset without compromising the main account credentials.

## Installation Steps

### 1. Install Rclone

Install the rclone package using pacman:

```bash
sudo pacman -S rclone
```

### 2. Configure Rclone for Cloudflare R2

Start the interactive configuration:

```bash
rclone config
```

Follow these steps in the configuration wizard:

#### Step 2.1: Create New Remote

```
e/n/d/r/c/s/q> n
```

#### Step 2.2: Name Your Remote

```
Enter name for new remote.
name> spark-perception
```

#### Step 2.3: Select Storage Type

Choose option **4** for Amazon S3 Compatible Storage:

```
Storage> 4
```

#### Step 2.4: Select Provider

Choose option **6** for Cloudflare R2:

```
provider> 6
```

#### Step 2.5: Environment Authentication

Select option **1** to enter credentials manually:

```
env_auth> 1
```

#### Step 2.6: Enter Access Credentials

You'll need to obtain these credentials from your project administrator:

```
access_key_id> [YOUR_ACCESS_KEY_HERE]
secret_access_key> [YOUR_SECRET_ACCESS_KEY_HERE]
```

> **Note**: Contact the project administrator to receive your access key ID, secret access key, and endpoint URL.

#### Step 2.7: Set Region

```
region> auto
```

#### Step 2.8: Set Endpoint

```
endpoint> [YOUR_R2_ENDPOINT_HERE]
```

The endpoint should look like: `https://[account-id].r2.cloudflarestorage.com`

#### Step 2.9: Skip Advanced Config

```
y/n> n
```

#### Step 2.10: Confirm Configuration

Review the configuration and confirm:

```
y/e/d> y
```

#### Step 2.11: Exit Configuration

```
e/n/d/r/c/s/q> q
```

### 3. Verify Configuration

List configured remotes to verify the setup:

```bash
rclone config show
```

You should see output similar to:

```
[spark-perception]
type = s3
provider = Cloudflare
access_key_id = ***
secret_access_key = ***
endpoint = ***
```

## Usage Options

### Option A: Mount Bucket as File System (Read-Only)

Mount the bucket to your local file system to browse it with your file manager (Dolphin, Nautilus, etc.):

**Create the mount point first:**

```bash
mkdir -p ~/spark-bucket
```

**Mount bucket:**

```bash
rclone mount spark-perception:apache-spark-perception-tree-rings ~/spark-bucket --read-only
```

**Important Notes:**

- This command will keep the terminal busy while the mount is active
- The bucket will be available at `~/spark-bucket` in your file manager
- It's mounted as **read-only** for safety
- Press `Ctrl+C` in the terminal to unmount

### Option B: Use Rclone Browser (GUI)

Rclone Browser provides a graphical interface for browsing and managing remote storage.

#### Install Rclone Browser

```bash
yay -S rclone-browser
```

#### Configure Alias for Browser

To use Rclone Browser effectively, create an alias pointing to the specific bucket:

1. Run rclone config again:

```bash
rclone config
```

2. Create a new remote (alias):

```
e/n/d/r/c/s/q> n
```

3. Name the alias:

```
Enter name for new remote.
name> alias-spark-bucket
```

4. Select storage type **3** (Alias):

```
Storage> 3
```

5. Set the remote path:

```
remote> spark-perception:apache-spark-perception-tree-rings
```

6. Skip advanced config:

```
y/n> n
```

7. Confirm and exit:

```
y/e/d> y
e/n/d/r/c/s/q> q
```

#### Launch Rclone Browser

```bash
rclone-browser
```

In the Rclone Browser interface:

1. Select `alias-spark-bucket` from the remotes list
2. Browse the bucket contents graphically
3. Download files as needed

## Adaptation for Other Linux Distributions

### Ubuntu/Debian

```bash
# Install rclone
sudo apt update
sudo apt install rclone

# Install rclone-browser (requires adding PPA or building from source)
# See: https://github.com/kapitainsky/RcloneBrowser/releases
```

### Fedora

```bash
# Install rclone
sudo dnf install rclone

# Install rclone-browser
sudo dnf install rclone-browser
```

### openSUSE

```bash
# Install rclone
sudo zypper install rclone

# Install rclone-browser (may require manual installation)
```

## Security Best Practices

1. **Never share your access credentials** in commits or public channels
2. **Use read-only mounts** when possible to prevent accidental modifications
3. **Keep your access tokens secure** - treat them like passwords
4. **Verify bucket contents** before downloading large datasets
5. **Contact the administrator** if you suspect credential compromise

## Troubleshooting

### "Failed to create file system" error

Ensure your credentials are correct:

```bash
rclone config show
rclone lsd spark-perception:apache-spark-perception-tree-rings
```

### Mount point busy or already in use

Unmount the existing mount:

```bash
fusermount -u ~/spark-bucket
```

### Rclone Browser doesn't show remotes

Verify that your alias is created correctly:

```bash
rclone config show
```

You should see both `spark-perception` and `alias-spark-bucket`.

### Permission denied when mounting

Install `fuse3` package:

```bash
sudo pacman -S fuse3
```

## Next Steps

Once Rclone is configured, you can:

- Browse the shared datasets in the bucket
- Download training data for local Spark applications
- Verify that uploaded results are stored correctly

Return to the main [README.md](README.md) to continue with the project setup.
