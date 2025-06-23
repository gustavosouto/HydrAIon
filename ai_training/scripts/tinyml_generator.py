#!/usr/bin/env python3
"""
Script simplificado para gerar modelo TinyML para o Projeto HydrAIon
Este script cria um modelo leve e otimizado para ESP32.

Autor: Gustavo Souto Silva de Barros Santos
Projeto: HydrAIon - Sistema IoT com IA para Monitoramento Inteligente da Qualidade da Água
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from datetime import datetime

def create_simple_model():
    """
    Cria um modelo muito simples para TinyML
    """
    model = keras.Sequential([
        layers.Input(shape=(5,)),  # pH, turbidez, OD, condutividade, temperatura
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 classes: Boa, Regular, Crítica
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_simple_data():
    """
    Gera dados sintéticos simples para treinamento rápido
    """
    np.random.seed(42)
    n_samples = 300
    
    X = []
    y = []
    
    for i in range(n_samples):
        if i < 100:  # Boa qualidade
            ph = np.random.normal(7.2, 0.2)
            turbidity = np.random.exponential(2.0)
            do = np.random.normal(8.0, 0.5)
            conductivity = np.random.normal(250, 30)
            temp = np.random.normal(24, 2)
            label = 0
        elif i < 200:  # Regular
            ph = np.random.normal(6.5, 0.5)
            turbidity = np.random.exponential(8.0)
            do = np.random.normal(6.0, 1.0)
            conductivity = np.random.normal(400, 50)
            temp = np.random.normal(26, 3)
            label = 1
        else:  # Crítica
            ph = np.random.normal(5.5, 0.8)
            turbidity = np.random.exponential(15.0)
            do = np.random.normal(3.0, 1.0)
            conductivity = np.random.normal(600, 100)
            temp = np.random.normal(28, 4)
            label = 2
        
        # Normalizar valores
        ph = np.clip(ph, 4.0, 10.0) / 10.0
        turbidity = np.clip(turbidity, 0.1, 50.0) / 50.0
        do = np.clip(do, 0.5, 12.0) / 12.0
        conductivity = np.clip(conductivity, 10, 1000) / 1000.0
        temp = np.clip(temp, 15, 35) / 35.0
        
        X.append([ph, turbidity, do, conductivity, temp])
        y.append(label)
    
    return np.array(X), np.array(y)

def convert_to_tflite_and_cpp(model, output_dir):
    """
    Converte modelo para TensorFlow Lite e gera arquivo C++
    """
    # Converter para TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Salvar modelo TFLite
    tflite_path = os.path.join(output_dir, 'hydraion_tinyml_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Modelo TensorFlow Lite salvo: {tflite_path}")
    print(f"Tamanho: {len(tflite_model) / 1024:.2f} KB")
    
    # Gerar arquivo C++ para ESP32
    model_data = ', '.join([str(b) for b in tflite_model])
    
    cpp_content = f"""// Modelo TinyML para o Projeto HydrAIon
// Gerado automaticamente em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#ifndef HYDRAION_TINYML_MODEL_H
#define HYDRAION_TINYML_MODEL_H

// Dados do modelo TensorFlow Lite
const unsigned char hydraion_tinyml_model[] = {{
{model_data}
}};

const int hydraion_tinyml_model_len = {len(tflite_model)};

// Configurações do modelo
const int HYDRAION_INPUT_SIZE = 5;
const int HYDRAION_OUTPUT_SIZE = 3;

// Classes de saída
const char* HYDRAION_CLASS_NAMES[] = {{"Boa", "Regular", "Crítica"}};

// Função para normalizar entradas
inline void normalize_inputs(float* inputs) {{
    inputs[0] = inputs[0] / 10.0f;  // pH (0-10)
    inputs[1] = inputs[1] / 50.0f;  // Turbidez (0-50 NTU)
    inputs[2] = inputs[2] / 12.0f;  // OD (0-12 mg/L)
    inputs[3] = inputs[3] / 1000.0f; // Condutividade (0-1000 µS/cm)
    inputs[4] = inputs[4] / 35.0f;  // Temperatura (0-35°C)
}}

#endif // HYDRAION_TINYML_MODEL_H
"""
    
    cpp_path = os.path.join(output_dir, 'hydraion_tinyml_model.h')
    with open(cpp_path, 'w') as f:
        f.write(cpp_content)
    
    print(f"Arquivo C++ gerado: {cpp_path}")

def main():
    """
    Função principal
    """
    print("=== Gerador de Modelo TinyML para HydrAIon ===")
    
    # Gerar dados sintéticos
    print("Gerando dados sintéticos...")
    X, y = generate_simple_data()
    
    # Criar modelo simples
    print("Criando modelo TinyML...")
    model = create_simple_model()
    model.summary()
    
    # Treinar rapidamente
    print("Treinamento rápido...")
    model.fit(X, y, epochs=50, batch_size=16, verbose=1)
    
    # Avaliar
    loss, accuracy = model.evaluate(X, y, verbose=0)
    print(f"Acurácia final: {accuracy:.4f}")
    
    # Converter e salvar
    output_dir = '../models'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Convertendo para TensorFlow Lite...")
    convert_to_tflite_and_cpp(model, output_dir)
    
    print("Modelo TinyML gerado com sucesso!")

if __name__ == "__main__":
    main()

