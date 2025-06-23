#!/usr/bin/env python3
"""
Script de treinamento de modelo de IA para o Projeto HydrAIon
Este script treina um modelo de classificação para qualidade da água usando TensorFlow/Keras.

Autor: Gustavo Souto Silva de Barros Santos
Projeto: HydrAIon - Sistema IoT com IA para Monitoramento Inteligente da Qualidade da Água
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import json
from datetime import datetime

class WaterQualityModelTrainer:
    """
    Classe para treinamento de modelos de classificação de qualidade da água
    """
    
    def __init__(self, model_name="hydraion_water_quality_v1"):
        self.model_name = model_name
        self.model = None
        self.history = None
        self.class_names = ['Boa', 'Regular', 'Crítica']
        
    def load_preprocessed_data(self, data_dir):
        """
        Carrega dados pré-processados
        
        Args:
            data_dir (str): Diretório com dados pré-processados
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
            X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
            X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
            y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
            y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
            y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
            
            print(f"Dados carregados:")
            print(f"  Treino: {X_train.shape}")
            print(f"  Validação: {X_val.shape}")
            print(f"  Teste: {X_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            print(f"Erro ao carregar dados pré-processados: {e}")
            return None
    
    def create_model(self, input_shape, num_classes=3):
        """
        Cria o modelo de rede neural
        
        Args:
            input_shape (tuple): Forma dos dados de entrada
            num_classes (int): Número de classes de saída
            
        Returns:
            tf.keras.Model: Modelo compilado
        """
        model = keras.Sequential([
            # Camada de entrada
            layers.Input(shape=input_shape),
            
            # Primeira camada densa com dropout
            layers.Dense(64, activation='relu', name='dense_1'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Segunda camada densa
            layers.Dense(32, activation='relu', name='dense_2'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Terceira camada densa
            layers.Dense(16, activation='relu', name='dense_3'),
            layers.Dropout(0.1),
            
            # Camada de saída
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Compilar o modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_lightweight_model(self, input_shape, num_classes=3):
        """
        Cria um modelo mais leve para TinyML
        
        Args:
            input_shape (tuple): Forma dos dados de entrada
            num_classes (int): Número de classes de saída
            
        Returns:
            tf.keras.Model: Modelo leve compilado
        """
        model = keras.Sequential([
            # Camada de entrada
            layers.Input(shape=input_shape),
            
            # Primeira camada densa menor
            layers.Dense(16, activation='relu', name='dense_1'),
            layers.Dropout(0.2),
            
            # Segunda camada densa
            layers.Dense(8, activation='relu', name='dense_2'),
            layers.Dropout(0.1),
            
            # Camada de saída
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Compilar o modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, X_train, X_val, y_train, y_val, epochs=100, batch_size=32, lightweight=False):
        """
        Treina o modelo
        
        Args:
            X_train, X_val: Dados de treino e validação
            y_train, y_val: Labels de treino e validação
            epochs (int): Número de épocas
            batch_size (int): Tamanho do batch
            lightweight (bool): Se deve usar modelo leve para TinyML
            
        Returns:
            tf.keras.Model: Modelo treinado
        """
        input_shape = (X_train.shape[1],)
        
        if lightweight:
            self.model = self.create_lightweight_model(input_shape)
            print("Criando modelo leve para TinyML...")
        else:
            self.model = self.create_model(input_shape)
            print("Criando modelo padrão...")
        
        print(f"Arquitetura do modelo:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Treinar o modelo
        print("Iniciando treinamento...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Treinamento concluído!")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Avalia o modelo no conjunto de teste
        
        Args:
            X_test: Dados de teste
            y_test: Labels de teste
            
        Returns:
            dict: Métricas de avaliação
        """
        if self.model is None:
            print("Erro: Modelo não foi treinado ainda.")
            return None
        
        # Avaliar no conjunto de teste
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Fazer predições
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Relatório de classificação
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'prediction_probabilities': y_pred_proba.tolist()
        }
        
        print(f"Avaliação do modelo:")
        print(f"  Acurácia no teste: {test_accuracy:.4f}")
        print(f"  Loss no teste: {test_loss:.4f}")
        print(f"\nRelatório de classificação:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return results
    
    def save_model(self, output_dir, include_tflite=True):
        """
        Salva o modelo treinado
        
        Args:
            output_dir (str): Diretório de saída
            include_tflite (bool): Se deve gerar versão TensorFlow Lite
        """
        if self.model is None:
            print("Erro: Modelo não foi treinado ainda.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Salvar modelo completo
        model_path = os.path.join(output_dir, f'{self.model_name}.h5')
        self.model.save(model_path)
        print(f"Modelo salvo em: {model_path}")
        
        # Salvar apenas os pesos
        weights_path = os.path.join(output_dir, f'{self.model_name}_weights.h5')
        self.model.save_weights(weights_path)
        print(f"Pesos salvos em: {weights_path}")
        
        # Salvar arquitetura como JSON
        architecture_path = os.path.join(output_dir, f'{self.model_name}_architecture.json')
        with open(architecture_path, 'w') as f:
            f.write(self.model.to_json())
        print(f"Arquitetura salva em: {architecture_path}")
        
        # Gerar versão TensorFlow Lite
        if include_tflite:
            self.convert_to_tflite(output_dir)
    
    def convert_to_tflite(self, output_dir):
        """
        Converte o modelo para TensorFlow Lite
        
        Args:
            output_dir (str): Diretório de saída
        """
        if self.model is None:
            print("Erro: Modelo não foi treinado ainda.")
            return
        
        # Converter para TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Otimizações para TinyML
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Quantização para reduzir tamanho
        converter.target_spec.supported_types = [tf.float16]
        
        # Converter
        tflite_model = converter.convert()
        
        # Salvar modelo TFLite
        tflite_path = os.path.join(output_dir, f'{self.model_name}.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Modelo TensorFlow Lite salvo em: {tflite_path}")
        print(f"Tamanho do modelo TFLite: {len(tflite_model) / 1024:.2f} KB")
        
        # Gerar arquivo C++ para ESP32
        self.generate_cpp_model(tflite_model, output_dir)
    
    def generate_cpp_model(self, tflite_model, output_dir):
        """
        Gera arquivo C++ com o modelo para ESP32
        
        Args:
            tflite_model: Modelo TensorFlow Lite
            output_dir (str): Diretório de saída
        """
        # Converter bytes para array C++
        model_data = ', '.join([str(b) for b in tflite_model])
        
        cpp_content = f"""// Modelo TensorFlow Lite para o Projeto HydrAIon
// Gerado automaticamente em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

#ifndef HYDRAION_MODEL_H
#define HYDRAION_MODEL_H

const unsigned char hydraion_model_data[] = {{
{model_data}
}};

const int hydraion_model_data_len = {len(tflite_model)};

// Classes de saída
const char* hydraion_class_names[] = {{"Boa", "Regular", "Crítica"}};
const int hydraion_num_classes = 3;

// Informações do modelo
const int hydraion_input_size = {self.model.input_shape[1]};
const int hydraion_output_size = 3;

#endif // HYDRAION_MODEL_H
"""
        
        cpp_path = os.path.join(output_dir, 'hydraion_model.h')
        with open(cpp_path, 'w') as f:
            f.write(cpp_content)
        
        print(f"Arquivo C++ gerado em: {cpp_path}")
    
    def plot_training_history(self, output_dir):
        """
        Plota o histórico de treinamento
        
        Args:
            output_dir (str): Diretório para salvar os gráficos
        """
        if self.history is None:
            print("Erro: Modelo não foi treinado ainda.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        
        # Plotar loss e accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Treino', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Validação', linewidth=2)
        ax1.set_title('Loss durante o Treinamento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Treino', linewidth=2)
        ax2.plot(self.history.history['val_accuracy'], label='Validação', linewidth=2)
        ax2.set_title('Acurácia durante o Treinamento')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico do histórico de treinamento salvo em: {output_dir}")
    
    def plot_confusion_matrix(self, cm, output_dir):
        """
        Plota a matriz de confusão
        
        Args:
            cm: Matriz de confusão
            output_dir (str): Diretório para salvar o gráfico
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Matriz de Confusão')
        plt.xlabel('Predição')
        plt.ylabel('Real')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matriz de confusão salva em: {output_dir}")
    
    def save_training_report(self, results, output_dir):
        """
        Salva relatório completo do treinamento
        
        Args:
            results (dict): Resultados da avaliação
            output_dir (str): Diretório de saída
        """
        report = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'model_architecture': {
                'layers': len(self.model.layers),
                'parameters': self.model.count_params(),
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape
            },
            'training_history': {
                'epochs_trained': len(self.history.history['loss']),
                'final_train_loss': float(self.history.history['loss'][-1]),
                'final_val_loss': float(self.history.history['val_loss'][-1]),
                'final_train_accuracy': float(self.history.history['accuracy'][-1]),
                'final_val_accuracy': float(self.history.history['val_accuracy'][-1])
            },
            'test_results': {
                'test_loss': float(results['test_loss']),
                'test_accuracy': float(results['test_accuracy']),
                'classification_report': results['classification_report']
            }
        }
        
        report_path = os.path.join(output_dir, f'{self.model_name}_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Relatório de treinamento salvo em: {report_path}")

def main():
    """
    Função principal para executar o treinamento
    """
    # Configurar TensorFlow
    tf.random.set_seed(42)
    
    # Verificar se GPU está disponível
    if tf.config.list_physical_devices('GPU'):
        print("GPU detectada e será utilizada para treinamento.")
    else:
        print("Treinamento será realizado na CPU.")
    
    # Carregar dados pré-processados
    data_dir = '../data/processed'
    trainer = WaterQualityModelTrainer()
    
    data = trainer.load_preprocessed_data(data_dir)
    if data is None:
        print("Erro: Não foi possível carregar os dados pré-processados.")
        print("Execute primeiro o script data_preprocessing.py")
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    # Treinar modelo padrão
    print("\n=== Treinando Modelo Padrão ===")
    trainer.train_model(X_train, X_val, y_train, y_val, epochs=100, lightweight=False)
    
    # Avaliar modelo
    results = trainer.evaluate_model(X_test, y_test)
    
    # Salvar modelo e resultados
    output_dir = '../models'
    trainer.save_model(output_dir, include_tflite=True)
    trainer.plot_training_history(output_dir)
    trainer.plot_confusion_matrix(np.array(results['confusion_matrix']), output_dir)
    trainer.save_training_report(results, output_dir)
    
    # Treinar modelo leve para TinyML
    print("\n=== Treinando Modelo Leve para TinyML ===")
    trainer_lite = WaterQualityModelTrainer("hydraion_water_quality_tinyml_v1")
    trainer_lite.train_model(X_train, X_val, y_train, y_val, epochs=100, lightweight=True)
    
    # Avaliar modelo leve
    results_lite = trainer_lite.evaluate_model(X_test, y_test)
    
    # Salvar modelo leve
    trainer_lite.save_model(output_dir, include_tflite=True)
    trainer_lite.plot_training_history(output_dir)
    trainer_lite.save_training_report(results_lite, output_dir)
    
    print("\n=== Treinamento Concluído ===")
    print(f"Modelo padrão - Acurácia: {results['test_accuracy']:.4f}")
    print(f"Modelo TinyML - Acurácia: {results_lite['test_accuracy']:.4f}")
    print(f"Modelos salvos em: {output_dir}")

if __name__ == "__main__":
    main()

