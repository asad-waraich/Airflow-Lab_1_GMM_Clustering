# Airflow Lab 1 - GMM Clustering Implementation

## 📋 Project Overview
This repository contains my implementation of Lab 1 for the MLOps course, focusing on Apache Airflow workflow orchestration with Gaussian Mixture Model (GMM) clustering. This is a modified version of the [original lab](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Airflow_Labs/Lab_1) by Professor Ramin Mohammadi.

## 🔧 My Modifications and Fixes

### 1. **Fixed Missing Python Dependencies**
The original lab had module import errors that prevented the DAG from running. I resolved these by adding the required packages to the Docker environment:

#### **Issue Encountered:**
```
ModuleNotFoundError: No module named 'cv2'
ModuleNotFoundError: No module named 'sklearn'
```

#### **Solution Implemented:**
Modified `docker-compose.yaml` to include essential data science packages:
```yaml
environment:
  _PIP_ADDITIONAL_REQUIREMENTS: |
    opencv-python-headless
    scikit-learn
    numpy
    pandas
    matplotlib
    scipy
```

### 2. **Implemented GMM Segmentation DAG**
Created a complete Airflow DAG (`gmm_segmentation.py`) that:
- Loads image data for processing
- Applies Gaussian Mixture Model clustering
- Performs image segmentation tasks
- Handles data serialization between tasks using XCom

### 3. **Environment Configuration**
- Set up proper working directories for data processing
- Configured Airflow to handle pickle serialization for model passing between tasks
- Added volume mounts for persistent data storage

### 4. **Docker Compose Optimizations**
Updated the `docker-compose.yaml` with:
- Proper memory allocation settings
- Volume mappings for working data
- Custom admin credentials configuration
- Disabled example DAGs for cleaner UI

## 🚀 How to Run

### Prerequisites
- Docker Desktop with at least 4GB RAM allocated (8GB recommended)
- Docker Compose installed

### Setup Steps

1. **Clone this repository:**
```bash
git clone https://github.com/asad-waraich/Airflow-Lab_1_GMM_Clustering.git
cd Airflow-Lab_1_GMM_Clustering
```

2. **Initialize Airflow:**
```bash
# Create necessary directories
mkdir -p ./dags ./logs ./plugins ./working_data

# Set Airflow user
echo -e "AIRFLOW_UID=$(id -u)" > .env

# Initialize the database
docker compose up airflow-init
```

3. **Start Airflow:**
```bash
docker compose up
```

4. **Access the Airflow UI:**
- Navigate to `http://localhost:8080`
- Login with:
  - Username: `airflow2`
  - Password: `airflow2`

5. **Run the GMM Segmentation DAG:**
- Find `gmm_segmentation` DAG in the UI
- Toggle it ON
- Click "Trigger DAG" to run

## 📁 Project Structure
```
Airflow_Lab_1_GMM_Clustering/
├── Lab_1/
│   ├── dags/
│   │   └── gmm_segmentation.py    # Main DAG file
│   ├── src/
│   │   └── lab.py                 # GMM clustering functions
│   └── data/
│       └── (input data files)
├── docker-compose.yaml             # Modified Docker configuration
├── .env                           # Environment variables
└── README.md                      # This file
```

## 🛠️ Technologies Used
- **Apache Airflow 2.5.1** - Workflow orchestration
- **Docker & Docker Compose** - Containerization
- **Python 3.7** - Programming language
- **Scikit-learn** - Machine learning library for GMM
- **OpenCV** - Image processing
- **NumPy & Pandas** - Data manipulation

## 📊 DAG Workflow

The GMM segmentation pipeline consists of the following tasks:

1. **load_data_task**: Loads image data from CSV/files
2. **data_preprocessing_task**: Preprocesses and normalizes the data
3. **build_save_model_task**: Trains GMM model and saves it
4. **load_model_task**: Loads the model and performs clustering/segmentation

Task Dependencies:
```
load_data → data_preprocessing → build_save_model → load_model
```

## 📊 Successful DAG Execution

The following screenshot shows the successful execution of the GMM Segmentation DAG with all tasks completed:

![Airflow DAG Success](assets/graph_view.png)

All tasks shown in green indicate successful completion:
- ✅ **start** - DAG initialization
- ✅ **run_gmm** - GMM model execution  
- ✅ **evaluate** - Model evaluation
- ✅ **visualize** - Results visualization
- ✅ **end** - DAG completion

## 🎨 GMM Segmentation Results

The pipeline successfully performs image segmentation using Gaussian Mixture Models. Here's an example output showing the segmented image with different regions identified by the GMM algorithm:

![GMM Segmentation Result](working_data/gmm_results/segmentation_2025-10-20.png)

The segmentation clearly identifies:
- **Green regions**: Background/vegetation areas
- **Pink/Magenta regions**: Primary subject (appears to be an animal)
- **Light green regions**: Ground/foreground areas

## 🔧 Additional Enhancements

### Visualization Function Added
I implemented a custom visualization function to save the segmentation results as PNG images for easier viewing:
```python
def visualize_results(**context):
    """Load and visualize the segmentation results"""
    import pickle
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for Docker
    import matplotlib.pyplot as plt
    import numpy as np
    
    ti = context['ti']
    segmentation_path = f'/opt/airflow/working_data/gmm_results/segmentation_{context["ds"]}.pkl'
    
    # Load segmentation image
    with open(segmentation_path, 'rb') as f:
        segmentation_image = pickle.load(f)
    
    # Save as PNG
    png_path = segmentation_path.replace('.pkl', '.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(segmentation_image)
    plt.title('GMM Segmentation Result')
    plt.axis('off')
    plt.savefig(png_path)
    plt.close()  # Properly close the figure to free memory
    
    print(f"Visualization saved to {png_path}")
    return f"Image saved as {png_path}"
```

This function:
- Loads the pickled segmentation results
- Creates a high-quality visualization using matplotlib
- Saves the result as a PNG file for easy viewing
- Properly manages memory by closing figures after saving

## 🐛 Troubleshooting

### Common Issues and Solutions:

1. **Import Errors**: Ensure all packages in `_PIP_ADDITIONAL_REQUIREMENTS` are properly installed
2. **Memory Issues**: Allocate at least 4GB to Docker Desktop
3. **Port Conflicts**: Ensure port 8080 is available
4. **Permission Errors**: Check the AIRFLOW_UID in .env matches your user ID

## 📝 Key Learnings
- Understanding DAG construction and task dependencies in Airflow
- Configuring Docker environments for data science workflows
- Handling data serialization between Airflow tasks
- Debugging and resolving Python package dependencies in containerized environments

## 🙏 Acknowledgments
- Original lab materials by [Professor Ramin Mohammadi](https://github.com/raminmohammadi)
- MLOps course at Northeastern University


---
**Author:** Asadullah Waraich  
**Course:** MLOps - Fall 2025  
**Institution:** Northeastern University