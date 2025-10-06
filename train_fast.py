"""
Fast Training Script for Object Detection
Optimized for quick training with your detection_dataset folder
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import logging
from pathlib import Path
import argparse
import os

from object_detection_model import (
    HomeObjectDetectionModel, 
    FastDetectionDataset, 
    get_fast_transforms,
    collate_fn,
    train_fast_detection_model,
    create_sample_detection_dataset,
    HOME_OBJECTS,
    NUM_CLASSES
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_training_environment():
    """Setup optimized training environment"""
    # Set environment variables for faster training
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # Set number of threads
    torch.set_num_threads(4)

def check_dataset(dataset_dir):
    """Check and prepare dataset"""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        logger.info(f"Creating dataset directory: {dataset_dir}")
        dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Count existing images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend(dataset_path.glob(f"*{ext}"))
        image_files.extend(dataset_path.glob(f"*{ext.upper()}"))
    
    logger.info(f"Found {len(image_files)} images in {dataset_dir}")
    
    # Create sample data if needed
    if len(image_files) < 50:
        logger.info("Insufficient training data. Creating sample dataset...")
        create_sample_detection_dataset(dataset_dir, num_images=200)
        
        # Recount
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(dataset_path.glob(f"*{ext}"))
            image_files.extend(dataset_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"Dataset now contains {len(image_files)} images")
    
    return len(image_files)

def create_optimized_model():
    """Create model with optimizations"""
    model = HomeObjectDetectionModel(num_classes=NUM_CLASSES)
    
    # Apply weight initialization for faster convergence
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    return model

def fast_train(
    dataset_dir="./detection_dataset",
    batch_size=16,
    num_epochs=10,
    learning_rate=0.002,
    save_path="best_detection_model.pth",
    resume_from=None
):
    """Fast training function with all optimizations"""
    
    logger.info("=" * 60)
    logger.info("FAST OBJECT DETECTION TRAINING")
    logger.info("=" * 60)
    
    # Setup environment
    setup_training_environment()
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Check and prepare dataset
    num_images = check_dataset(dataset_dir)
    if num_images == 0:
        logger.error("No images found in dataset!")
        return
    
    # Create model
    model = create_optimized_model()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from and Path(resume_from).exists():
        logger.info(f"Resuming training from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Prepare data
    logger.info("Preparing datasets...")
    train_transform, val_transform = get_fast_transforms()
    
    # Use same dataset for train/val for speed (in production, split properly)
    train_dataset = FastDetectionDataset(dataset_dir, transform=train_transform, cache_images=True)
    val_dataset = FastDetectionDataset(dataset_dir, transform=val_transform, cache_images=False)
    
    # Optimized data loaders
    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn
    )
    
    logger.info(f"Training setup:")
    logger.info(f"  Dataset size: {len(train_dataset)}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Batches per epoch: {len(train_loader)}")
    logger.info(f"  Total epochs: {num_epochs}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Workers: {num_workers}")
    
    # Train the model
    start_time = time.time()
    
    try:
        best_loss = train_fast_detection_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            save_path=save_path
        )
        
        training_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Total training time: {training_time/60:.1f} minutes")
        logger.info(f"Best loss achieved: {best_loss:.4f}")
        logger.info(f"Model saved as: {save_path}")
        
        # Save additional formats for API compatibility
        model_state = torch.load(save_path, map_location='cpu')
        if isinstance(model_state, dict) and 'model_state_dict' in model_state:
            state_dict = model_state['model_state_dict']
        else:
            state_dict = model_state
        
        # Save in different formats
        torch.save(state_dict, 'home_objects_detection_model.pth')
        torch.save(state_dict, 'home_objects_cnn.pth')
        
        logger.info("Model saved in multiple formats for API compatibility")
        
        # Test the trained model
        logger.info("Testing trained model...")
        model.eval()
        test_input = torch.randn(1, 3, 416, 416).to(device)
        
        with torch.no_grad():
            test_start = time.time()
            test_output = model(test_input)
            test_time = time.time() - test_start
            
            # Test prediction
            detections = model.predict(test_input, conf_thresh=0.1)
            
        logger.info(f"Model test successful:")
        logger.info(f"  Inference time: {test_time*1000:.1f}ms")
        logger.info(f"  Output shape: {test_output.shape}")
        logger.info(f"  Test detections: {len(detections[0])}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Fast Object Detection Training")
    
    parser.add_argument('--dataset', type=str, default='./detection_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.002,
                       help='Learning rate')
    parser.add_argument('--save-path', type=str, default='best_detection_model.pth',
                       help='Path to save the trained model')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--quick', action='store_true',
                       help='Quick training mode (5 epochs, smaller batch)')
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.epochs = 5
        args.batch_size = min(args.batch_size, 8)
        logger.info("Quick training mode enabled")
    
    # Run training
    success = fast_train(
        dataset_dir=args.dataset,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        save_path=args.save_path,
        resume_from=args.resume
    )
    
    if success:
        logger.info("Training completed successfully!")
        logger.info("You can now run the API server with: python api_server.py")
    else:
        logger.error("Training failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())