# Apache Spark Distributed Computing Environment

This project sets up a distributed Apache Spark cluster using Docker, along with a Python development environment for running Spark applications.

## Prerequisites

- Docker and Docker Compose installed
- **Python 3.10.19** (exact version required to match Docker containers)
  - **For Arch Linux/Manjaro users**: See [PYTHON_SETUP_GUIDE.md](PYTHON_SETUP_GUIDE.md) for detailed installation instructions
  - This guide can be adapted for other Linux distributions
- Git

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd apache-spark
```

### 2. Install Python Dependencies

Create a virtual environment and install the required libraries:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

The `requirements.txt` includes:

- **PySpark 4.0.1**: Apache Spark Python API
- **Jupyter**: For interactive notebooks
- **IPython**: Enhanced Python shell
- **Pre-commit**: Git hooks for code quality
- And other supporting libraries

### 3. Setup Conventional Commits

Run the setup script to configure pre-commit hooks for conventional commits:

```bash
chmod +x setup.sh
./setup.sh
```

This script will:

- Create a virtual environment (if not already created)
- Install all project dependencies
- Install and configure pre-commit hooks
- Set up commit message validation (enforces `feat:`, `fix:`, `docs:`, etc. format)

#### Code Quality (Optional)

The project includes Black and Flake8 linters (currently disabled in pre-commit). To run them manually:

```bash
source env/bin/activate
black src/  # Format code
flake8 src/  # Check code style
```

To enable automatic linting on commit, uncomment the linter sections in `.pre-commit-config.yaml`

### 4. Start the Spark Cluster

Launch the Spark cluster with 1 master and 3 workers using Docker Compose:

```bash
docker compose up --scale spark-worker=3
```

This will start:

- **1 Spark Master** (accessible at `http://localhost:8080`)
- **3 Spark Workers** (each with 2 cores and 2GB memory)

To run in detached mode:

```bash
docker compose up --scale spark-worker=3 -d
```

### 5. Verify the Cluster

Once the cluster is running, you can access the Spark Master UI at:

```
http://localhost:8080
```

You should see 3 workers registered with the master.

## Running Spark Applications

### Using Jupyter Notebooks

1. Activate the virtual environment:

   ```bash
   source env/bin/activate
   ```

2. Open the test notebook at `notebooks/test_spark.ipynb`

3. The notebook demonstrates:
   - Connecting to the Spark cluster
   - Running distributed tasks
   - Comparing sequential vs distributed execution

### Connection Configuration

When connecting to the Spark cluster from your local machine, use:

- **Master URL**: `spark://localhost:7077`
- **Driver Host IP**: Your Docker bridge IP (typically `172.26.0.1`)

Example connection code:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("spark://localhost:7077") \
    .appName("MyApp") \
    .config("spark.driver.host", "172.26.0.1") \
    .getOrCreate()
```

## Stopping the Cluster

To stop the Spark cluster:

```bash
docker compose down
```

## Project Structure

```
.
├── docker-compose.yml      # Spark cluster configuration
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script for pre-commit hooks
├── env/                   # Python virtual environment
└── notebooks/             # Jupyter notebooks
    └── test_spark.ipynb   # Example Spark application
```

## Troubleshooting

### Workers not connecting to master

- Check that the master is fully started before workers attempt to connect
- Verify port 7077 is accessible
- Check Docker logs: `docker compose logs`

### Driver connection issues

- Ensure the `spark.driver.host` matches your Docker bridge IP
- Find your Docker bridge IP: `ip addr show docker0` or `ifconfig docker0`
