#!/bin/bash

# Monitor training progress

LOG_FILE="/Users/louis/Desktop/yearbook-urap/layout-model-training/training.log"

echo "=========================================="
echo "Training Monitor"
echo "=========================================="
echo ""

# Check if training is running
if ps aux | grep -v grep | grep "train_net.py" > /dev/null; then
    echo "✅ Training is RUNNING"
    echo ""
else
    echo "❌ Training is NOT running"
    echo ""
fi

# Show recent progress
echo "Recent training output:"
echo "----------------------------------------"
tail -30 "$LOG_FILE" | grep -E "eta:|loss_cls|loss_box|total_loss|bbox/AP|iteration" || tail -30 "$LOG_FILE"
echo "----------------------------------------"
echo ""

# Check for completion
if grep -q "Total training time" "$LOG_FILE"; then
    echo "🎉 TRAINING COMPLETE!"
    echo ""
    echo "Model saved to:"
    echo "  layout-model-training/outputs/yearbook/fast_rcnn_R_50_FPN_3x/model_final.pth"
    echo ""
    echo "Next: Test your model with:"
    echo "  cd layout-model-training"
    echo "  python3 test_trained_model.py --input /path/to/test_page.jpg"
else
    echo "⏳ Training in progress..."
    echo ""
    echo "To check progress again, run: bash monitor_training.sh"
    echo "To view full log: tail -f $LOG_FILE"
fi
