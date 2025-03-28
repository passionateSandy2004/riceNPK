import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 1  # Use single worker to avoid memory issues
worker_class = "sync"
worker_connections = 1000
timeout = 300  # Increase timeout to 5 minutes
keepalive = 5
max_requests = 1000
max_requests_jitter = 50
worker_tmp_dir = "/dev/shm"  # Use shared memory for temporary files
preload_app = True  # Preload the application to share memory between workers

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "rice_npk_predictor"

# Memory management
max_worker_lifetime = 3600  # Restart workers after 1 hour 
