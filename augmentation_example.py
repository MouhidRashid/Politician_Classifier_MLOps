# Data Augmentation Example for Training
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,           # rotation
    width_shift_range=0.2,      # cropping/translation
    height_shift_range=0.2,     # cropping/translation
    shear_range=0.2,
    zoom_range=0.2,             # zooming
    horizontal_flip=True,       # flipping
    brightness_range=[0.8,1.2], # brightness variation
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# For validation and test, use only rescale:
val_test_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_test_datagen.flow_from_directory(
    'dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
test_generator = val_test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
