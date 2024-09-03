import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

def segment_image(image):
    # Görüntü işleme
    if len(image.shape) == 2:  # Eğer gri tonlamalı ise
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image  # Zaten RGB formatında

    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    thresh_value, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(image_rgb)
    ax[0].set_title('Orijinal Görüntü (RGB)')
    ax[0].axis('off')
    
    ax[1].imshow(binary_image, cmap='gray')
    ax[1].set_title('Segmentasyon Sonucu (Eşikleme)')
    ax[1].axis('off')
    
    plt.tight_layout()
    
    # Geçici dosya oluşturma
    result_path = tempfile.mktemp(suffix='.png')
    plt.savefig(result_path)
    plt.close()
    
    return result_path

# Gradio arayüzü
interface = gr.Interface(
    fn=segment_image,  # Segmentasyon fonksiyonu
    inputs=gr.Image(type="numpy", label="Resim Yükleyin"),  # Tek resim girişi
    outputs=gr.Image(label="Segmentasyon Sonucu"),  # Segmentasyon sonucunu gösterecek widget
    title="Görüntü Segmentasyonu",  # Arayüz başlığı
    description="Bir resim yükleyin ve Otsu eşikleme yöntemiyle segmentasyon sonucunu görün."  # Açıklama
)

# Arayüzü başlatma
interface.launch()