import numpy as np

import cv2
import albumentations as A
import torch
import torch.nn as nn
import timm

import tqdm
import joblib
from torchvision import  models

class MultiTaskModel(nn.Module):
    def __init__(self, num_classes_multiclass):
        super(MultiTaskModel, self).__init__()
        
        # Загружаем предобученную модель EfficientNet-B0

        #self.backbone = timm.create_model('efficientnetv2_rw_t', pretrained=True)
        self.backbone = models.efficientnet_v2_s(weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1")
        #self.backbone = models.efficientnet_b1(weights="EfficientNet_B1_Weights.DEFAULT")
        # Изменяем последний слой, чтобы использовать его как общий экстрактор признаков
        num_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Identity()  # Убираем классификационный слой
        
        # Головной слой для бинарной классификации
        self.binary_classifier = nn.Sequential(
            nn.Linear(num_features, 1),
            #nn.Sigmoid()  # Используем сигмоид для бинарной классификации
        )
        
        # Головной слой для многоклассовой классификации
        self.multiclass_classifier = nn.Sequential(
            nn.Linear(num_features, num_classes_multiclass)
            # Используем Softmax при вычислении потерь, так что здесь активация не нужна
        )

    def forward(self, x):
        # Пропускаем изображение через EfficientNet и получаем признаки
        features = self.backbone(x)
        
        # Пропускаем признаки через бинарный и многоклассовый классификаторы
        binary_output = self.binary_classifier(features)
        multiclass_output = self.multiclass_classifier(features)
        
        return binary_output, multiclass_output


class FinalModel(nn.Module):
    def __init__(self, num_classes_multiclass):
        super(FinalModel, self).__init__()
        self.backbone = MultiTaskModel(num_classes_multiclass)
        self.backbone.load_state_dict(torch.load("weights_layer_group0_epoch_3.pth", weights_only=True))
        # with open("./calibrators/binary_calibrator.sav", 'rb') as f:
        #     self.binary_calibrator = joblib.load(f)
        self.binary_calibrator = joblib.load("./calibrators/binary_calibrator (2).sav")
        self.bin_classifier = models.efficientnet_b0()
        self.bin_classifier.classifier[1] = nn.Linear(1280, 1, bias=True)
        self.bin_classifier.load_state_dict(torch.load("weights_layer_group0_epoch_3_bin_b0.pth", weights_only=True))
        self.names = ["Melanoma", "Melanocytic nevus", "Basal cell carcinoma", "Actinic keratosis", "Benign keratosis (solar lentigo / seborrheic keratosis / lichen planus-like keratosis)", "Squamous cell carcinoma"]
        # for i in range(num_classes_multiclass):
        #     # with open("./calibrators/multiclass_calibrator.sav", 'rb') as f:
        #     #     self.multiclass_calibrators.append(joblib.load(f))
        #     self.multiclass_calibrators.append(joblib.load(f"./calibrators/multiclass_calibrator_{i}.sav"))
    
    def preprocess_image(self, image_path):
        """Функция для чтения и предобработки изображения."""
        transform = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))  # Изменение размера до (224, 224)
        image = transform(image=image)["image"]
        image = np.transpose(image, (2, 0, 1))  # Меняем порядок осей на (C, H, W)
        image = torch.tensor(image).unsqueeze(0)  # Преобразуем в тензор и добавляем измерение батча
        return image
    
    def calibrated_predict(self, image_paths):
        """
        Функция для предсказания с учетом калибровки.
        
        Параметры:
        - model: обученная модель PyTorch с двумя выходами (бинарный и многоклассовый).
        - binary_calibrator: калибровщик для бинарного классификатора.
        - multiclass_calibrators: список калибровщиков для многоклассового классификатора.
        - image_paths: список путей к изображениям.
        
        Возвращает:
        - Словарь с предсказаниями вероятностей и метками для бинарного и многоклассового классификаторов.
        """
        self.backbone.eval()  # Перевод модели в режим инференса
        
        binary_probs = []
        binary_preds = []
        multiclass_probs = []
        multiclass_preds = []
        if type(image_paths) is not list:
            image_paths = [image_paths]
        with torch.no_grad():
            for image_path in tqdm.tqdm(image_paths, desc="Processed"):
                # Чтение и предобработка изображения
                image = self.preprocess_image(image_path)
                
                # Прогон через модель
                binary_logits, multiclass_logits = self.backbone(image)
                
                # Преобразование логитов в numpy для использования в калибраторе
                binary_logits_np = binary_logits.numpy().reshape(-1, 1)
                #multiclass_logits_np = multiclass_logits.numpy()
                
                # Применение калибровки к бинарному классификатору
                #binary_prob = self.binary_calibrator.predict_proba(binary_logits_np)[:, 1]
                binary_prob_1 = torch.sigmoid(self.bin_classifier(image).squeeze(1)).numpy()
                binary_prob_2 = torch.sigmoid(binary_logits.squeeze(1)).numpy()
                binary_prob = ((3*binary_prob_1 + 3*binary_prob_2) / 2)[0]
                #binary_pred = (binary_prob >= 0.5).astype(int)  # Предсказание при пороге 0.5
                # Применение калибровки к многоклассовому классификатору
                # multiclass_prob = np.zeros_like(multiclass_logits_np)
                # for class_idx, calibrator in enumerate(self.multiclass_calibrators):
                #     multiclass_prob[:, class_idx] = calibrator.predict_proba(multiclass_logits_np[:, [class_idx]])[:, 1]
                # multiclass_pred = np.argmax(multiclass_prob, axis=1)
                # Сохранение результатов
                multiclass_prob = torch.softmax(multiclass_logits, dim=1).numpy()
                print(multiclass_prob)
                class_dict = {name: multiclass_prob[0][i] for i, name in enumerate(['MEL', 'NV', 'BCC', 'AK', 'BKL', 'SCC'])}
                # binary_prob = 0.0
                # for mal in ["MEL", "BCC", "SCC", "AK"]:
                #     binary_prob += class_dict[mal]
                multiclass_pred = np.argmax(multiclass_prob, axis=1)
                binary_probs.append(binary_prob)
                #binary_preds.append(binary_pred[0])
                multiclass_probs.append(np.max(multiclass_prob[0]))
                multiclass_preds.append(multiclass_pred[0])
        multiclass_preds = [self.names[int(i)] for i in multiclass_preds]
        return {
            "binary_probs": binary_probs,
            "multiclass_probs": multiclass_probs,
            "multiclass_preds": multiclass_preds
        }
