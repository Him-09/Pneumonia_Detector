# Pneumonia Detection from Chest X-Ray Images

This project uses deep learning to classify chest X-ray images as **normal** or **pneumonia**. It leverages transfer learning with ResNet18 and provides scripts for training, evaluation, and data handling.

---

## Project Structure

```
Pneu/
├── src/
│   ├── evaluate.py
│   ├── train.py
│   ├── data/
│   │   └── dataset.py
│   ├── img_data/                # Folder with X-ray images
│   ├── labeled_data.csv         # Full dataset labels
│   ├── test_labeled_data.csv    # Test set labels
│   └── ...
├── best_model.pth               # Saved trained model
├── training_history.csv         # Training/validation metrics
└── README.md
```

---

## Dataset

- **labeled_data.csv**: Contains image filenames and labels (`normal` or `pneumonia`).
- **img_data/**: Directory with all X-ray images.
- **test_labeled_data.csv**: Subset for testing.

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas
- numpy

Install dependencies:
```bash
pip install torch torchvision pandas numpy
```

---

## Training

Train the model (example, adjust as needed):

```bash
python src/train.py
```

- Uses transfer learning with ResNet18.
- Saves the best model as `best_model.pth`.
- Logs training history to `training_history.csv`.

---

## Evaluation

Evaluate the trained model on the test set:

```bash
python src/evaluate.py
```

- Loads `best_model.pth`.
- Prints test accuracy.

---

## Data Preparation

- Place all images in `src/img_data/`.
- Ensure `labeled_data.csv` and `test_labeled_data.csv` are in `src/`.
- Each CSV should have columns: `img_name,label`.

---

## Improving the Model

- Use data augmentation for better generalization.
- Tune hyperparameters (learning rate, batch size, etc.).
- Try different architectures (e.g., EfficientNet, DenseNet).
- Add regularization (dropout, weight decay).
- Analyze misclassifications for further insights.

---

## Results

- Example test accuracy: **87.34%**
- See `training_history.csv` for detailed metrics.

---

## License

This project is for educational and research purposes.

---

## Contact

For questions or contributions, please open an issue or pull request.
