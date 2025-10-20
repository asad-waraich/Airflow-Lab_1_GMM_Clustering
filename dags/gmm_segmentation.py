from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os

default_args = {
    'owner': 'asad_waraich',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def run_gmm_segmentation(**context):
    """
    Run the complete GMM segmentation pipeline
    """
    import cv2
    import numpy as np
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import KFold
    from sklearn.cluster import KMeans
    import pickle
    import os
    
    print("Starting GMM Image Segmentation...")
    
    # Load image
    image_path = '/opt/airflow/dags/data/TestImageHorse.jpg'

    if os.path.exists(image_path):
        print(f"Loading image from {image_path}")
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image_rgb.shape
        print(f"Image loaded successfully: {height}x{width}")
    else:
        print(f"Image not found at {image_path}, using synthetic data...")
        # Fallback to synthetic data if image not found
        height, width = 100, 100
        np.random.seed(42)
        image_rgb = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Extract features
    features = []
    for row in range(height):
        for col in range(width):
            red, green, blue = image_rgb[row, col]
            features.append([row, col, red, green, blue])
    
    features = np.array(features)
    
    # Normalize features
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)
    
    print(f"Image shape: {height}x{width}")
    print(f"Total pixels: {len(features)}")
    
    # Cross-validation for K selection
    k_values = range(2, min(8, len(features) // 100))  # Reduced range for faster execution
    num_folds = 3  # Reduced folds for faster execution
    
    log_likelihoods = []
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    print("Performing cross-validation...")
    for k in k_values:
        fold_scores = []
        for fold_idx, (train_indices, test_indices) in enumerate(kf.split(features_normalized)):
            train_data = features_normalized[train_indices]
            test_data = features_normalized[test_indices]
            
            # K-means initialization
            kmeans = KMeans(n_clusters=k, n_init=3, max_iter=50, random_state=42)
            kmeans.fit(train_data)
            
            # GMM fitting
            gmm = GaussianMixture(n_components=k, covariance_type='full', max_iter=50, random_state=42)
            gmm.means_init = kmeans.cluster_centers_
            gmm.fit(train_data)
            
            # Evaluate
            log_likelihood = gmm.score(test_data)
            fold_scores.append(log_likelihood)
            log_likelihoods.append((k, fold_idx + 1, log_likelihood))
        
        avg_score = np.mean(fold_scores)
        print(f"K={k}: Average log-likelihood = {avg_score:.4f}")
    
    # Determine optimal K
    log_likelihoods = np.array(log_likelihoods)
    
    # Calculate average log likelihood for each K
    average_scores = []
    for k in k_values:
        k_scores = log_likelihoods[log_likelihoods[:, 0] == k][:, 2]
        average_scores.append(np.mean(k_scores))
    
    # Find best K (85% threshold)
    if average_scores:
        asymptotic_value = max(average_scores)
        threshold = asymptotic_value * 0.85
        best_k = next((k for k, score in zip(k_values, average_scores) if score >= threshold), k_values[-1])
    else:
        best_k = 4  # Default
    
    print(f"\nOptimal K determined: {best_k}")
    
    # Train final GMM
    print(f"Training final GMM with K={best_k}...")
    final_gmm = GaussianMixture(n_components=best_k, covariance_type='full', max_iter=100, random_state=42)
    final_gmm.fit(features_normalized)
    
    # Perform segmentation
    labels = final_gmm.predict(features_normalized)
    segmentation_labels = labels.reshape(height, width)
    
    # Generate segmentation image
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(best_k, 3))
    segmentation_image_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    for label_index in range(best_k):
        segmentation_image_rgb[segmentation_labels == label_index] = colors[label_index]
    
    # Calculate statistics
    segment_stats = {}
    for i in range(best_k):
        pixel_count = np.sum(labels == i)
        percentage = (pixel_count / len(labels)) * 100
        segment_stats[f'segment_{i}'] = {
            'pixels': int(pixel_count),
            'percentage': round(percentage, 2)
        }
    
    # Save results
    output_dir = '/opt/airflow/working_data/gmm_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f'gmm_model_{context["ds"]}.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(final_gmm, f)
    
    # Save segmentation
    segmentation_path = os.path.join(output_dir, f'segmentation_{context["ds"]}.pkl')
    with open(segmentation_path, 'wb') as f:
        pickle.dump(segmentation_image_rgb, f)
    
    # Push results to XCom
    context['ti'].xcom_push(key='best_k', value=best_k)
    context['ti'].xcom_push(key='segment_stats', value=segment_stats)
    context['ti'].xcom_push(key='model_path', value=model_path)
    
    print("\nSegmentation Statistics:")
    for segment, stats in segment_stats.items():
        print(f"  {segment}: {stats['pixels']} pixels ({stats['percentage']}%)")
    
    return f"GMM segmentation complete with K={best_k}"


def evaluate_results(**context):
    """
    Evaluate and report on segmentation results
    """
    ti = context['ti']
    
    best_k = ti.xcom_pull(task_ids='run_gmm', key='best_k')
    segment_stats = ti.xcom_pull(task_ids='run_gmm', key='segment_stats')
    model_path = ti.xcom_pull(task_ids='run_gmm', key='model_path')
    
    print("=" * 50)
    print("GMM SEGMENTATION EVALUATION REPORT")
    print("=" * 50)
    print(f"Execution Date: {context['ds']}")
    print(f"Optimal Components: {best_k}")
    print(f"Model saved at: {model_path}")
    print("\nSegment Distribution:")
    
    for segment, stats in segment_stats.items():
        print(f"  {segment}: {stats['pixels']} pixels ({stats['percentage']}%)")
    
    # Quality assessment
    percentages = [stats['percentage'] for stats in segment_stats.values()]
    max_percentage = max(percentages)
    min_percentage = min(percentages)
    
    print("\nQuality Metrics:")
    print(f"  Largest segment: {max_percentage}%")
    print(f"  Smallest segment: {min_percentage}%")
    print(f"  Balance ratio: {min_percentage/max_percentage:.2f}")
    
    if max_percentage > 70:
        print("  Warning: One segment dominates the image")
    elif min_percentage < 5:
        print("  Warning: Very small segments detected")
    else:
        print("  Segmentation appears well-balanced")
    
    return "Evaluation complete"
def visualize_results(**context):
    """Load and visualize the segmentation results"""
    import pickle
    import matplotlib
    matplotlib.use('Agg') 
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
    plt.close()  # ADD THIS LINE
    
    print(f"Visualization saved to {png_path}")
    return f"Image saved as {png_path}"

# Create DAG
dag = DAG(
    'gmm_segmentation',
    default_args=default_args,
    description='Simple GMM image segmentation based on Asad\'s project',
    schedule_interval=None,
    catchup=False,
    tags=['gmm', 'computer-vision', 'ml-project']
)

# Define tasks
start = BashOperator(
    task_id='start',
    bash_command='echo "Starting GMM Segmentation Pipeline..."',
    dag=dag
)

gmm_task = PythonOperator(
    task_id='run_gmm',
    python_callable=run_gmm_segmentation,
    dag=dag
)

evaluate = PythonOperator(
    task_id='evaluate',
    python_callable=evaluate_results,
    dag=dag
)
visualize = PythonOperator(
    task_id='visualize',
    python_callable=visualize_results,
    dag=dag
)

end = BashOperator(
    task_id='end',
    bash_command='echo "GMM Pipeline Complete. Check working_data/gmm_results for outputs"',
    dag=dag
)

# Set dependencies
start >> gmm_task >> evaluate >> visualize >> end
