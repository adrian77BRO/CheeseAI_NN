from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
data_dir = Path('dataset')
class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
print("Clases encontradas:", class_names)

IMG_HEIGHT, IMG_WIDTH = 224, 224
images, labels = [], []

for class_name in class_names:
    class_folder = data_dir / class_name
    for img_path in class_folder.glob('*'):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(class_to_idx[class_name])
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")

X = np.array(images)
y = np.array(labels)
print("Total de imágenes:", len(X))

# Definir modelo
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Validación cruzada
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold = 1
accuracies = []
all_true = []
all_pred = []
best_accuracy = 0.0
best_model = None

# Para graficar pérdida
fold_train_losses = []
fold_val_losses = []

for train_index, test_index in kf.split(X):
    print(f"\nEntrenando Fold {fold}...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = create_model()

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )

    # Guardar historial de pérdidas
    fold_train_losses.append(history.history['loss'])
    fold_val_losses.append(history.history['val_loss'])

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Fold {fold} - Precisión en test: {test_acc:.4f}")
    accuracies.append(test_acc)

    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    all_true.extend(y_test)
    all_pred.extend(y_pred)

    if test_acc > best_accuracy:
        best_accuracy = test_acc
        best_model = model
        best_model.save('mejor_modelo.h5')
        print(f"Nuevo mejor modelo guardado con precisión: {best_accuracy:.4f}")

    fold += 1

# Resultados finales
avg_accuracy = np.mean(accuracies)
print(f"\nPrecisión promedio en {n_splits} folds: {avg_accuracy:.4f}")

cm = confusion_matrix(all_true, all_pred)
print("\nMatriz de Confusión:")
print(cm)

report = classification_report(all_true, all_pred, target_names=class_names)
print("\nReporte de Clasificación:")
print(report)

# Matriz de Confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.tight_layout()
plt.show()

# Gráfica de precisión por fold
plt.figure(figsize=(7, 4))
plt.plot(range(1, n_splits + 1), accuracies, marker='o', label='Precisión')
plt.axhline(avg_accuracy, color='red', linestyle='--', label=f'Promedio: {avg_accuracy:.4f}')
plt.title('Precisión por Fold')
plt.xlabel('Fold')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfica de pérdida promedio por época
max_epochs = max(len(losses) for losses in fold_train_losses)

def pad_losses(losses, max_len):
    return losses + [None] * (max_len - len(losses))

padded_train = np.array([pad_losses(l, max_epochs) for l in fold_train_losses])
padded_val = np.array([pad_losses(l, max_epochs) for l in fold_val_losses])

avg_train_loss = np.nanmean(np.where(padded_train != None, padded_train, np.nan), axis=0)
avg_val_loss = np.nanmean(np.where(padded_val != None, padded_val, np.nan), axis=0)

plt.figure(figsize=(8, 5))
plt.plot(range(1, max_epochs + 1), avg_train_loss, label='Pérdida de Entrenamiento', color='blue')
plt.plot(range(1, max_epochs + 1), avg_val_loss, label='Pérdida de Validación', color='orange')
plt.title('Pérdida Promedio por Época (Validación Cruzada)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()