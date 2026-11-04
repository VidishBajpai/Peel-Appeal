# Peel-Appeal: Fruit Freshness Classifier

A PyTorch image classifier that tells fresh from rotten fruit (8 fruits) and
rejects non-fruit images, with a Gradio web app.

![demo](screenshots/fruit.png) ![non-fruit](screenshots/notfruit.png)

## Results

| Model | Classes | Test accuracy |
|---|---|---|
| Baseline CNN (from scratch) | 16 | 83.3% |
| ResNet18, fine-tuned | 16 | 96.0% |
| ResNet18, fine-tuned + NotFruit class | 17 | 97.7% |

Fresh-vs-rotten accuracy (ignoring fruit type) for the 16-class ResNet: 98.5%.
The 17-class model uses a different split, so its number is comparable but not identical.

## Method
- Data: Mendeley "Fresh and Rotten Fruits Dataset" (original images, 200 per class).
  Stratified 70/15/15 split, test set used once.
- Transfer learning with ImageNet-pretrained ResNet18, Adam, 8 epochs, on Apple M2 (MPS).
- NotFruit class: 300 Imagenette images.
- Grad-CAM was used to check the model looks at the fruit, not the background.

## Known limitations (tested honestly)
- **Domain gap:** the dataset is close-up photos. On my own phone photos (9 fresh fruit,
  taken from farther away on a plain surface) accuracy was only [1/9]. The dataset's
  test score does not transfer to casual photos.
- Zooming in on the center helped partly (fresh-vs-rotten 3/9 to 6/9), but my test photos
  were all fresh, so this may only reflect a shift in bias.
- NotFruit was trained on 300 generic images; [describe what the shoe, robot, mug did].
- Only 9 own photos were tested, so that number is a rough indication, not a benchmark.
- Not for real food-safety decisions.

## Run it
    pip install -r requirements.txt
    cd app && python app.py

## Credits
Dataset: [add title, authors and license from the Mendeley page].
NotFruit images: Imagenette (fast.ai) [check license].