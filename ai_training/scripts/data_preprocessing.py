#!/usr/bin/env python3
"""
Script de pré-processamento de dados para o Projeto HydrAIon
Este script processa dados de sensores de qualidade da água e prepara datasets para treinamento de IA.

Autor: Gustavo Souto Silva de Barros Santos
Projeto: HydrAIon - Sistema IoT com IA para Monitoramento Inteligente da Qualidade da Água
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class WaterQualityPreprocessor:
    """
    Classe para pré-processamento de dados de qualidade da água
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = ['ph', 'turbidity', 'dissolved_oxygen', 'conductivity', 'temperature']
        self.target_column = 'water_quality_class'
        
    def load_data_from_firebase_export(self, file_path):
        """
        Carrega dados exportados do Firebase Realtime Database
        
        Args:
            file_path (str): Caminho para o arquivo JSON exportado do Firebase
            
        Returns:
            pd.DataFrame: DataFrame com os dados processados
        """
        try:
            with open(file_path, 'r') as f:
                firebase_data = json.load(f)
            
            # Processar dados do Firebase
            processed_data = []
            
            if 'devices' in firebase_data:
                for device_id, device_data in firebase_data['devices'].items():
                    if 'readings' in device_data:
                        for timestamp, reading in device_data['readings'].items():
                            row = {
                                'device_id': device_id,
                                'timestamp': int(timestamp),
                                'datetime': datetime.fromtimestamp(int(timestamp) / 1000),
                                **reading
                            }
                            processed_data.append(row)
            
            df = pd.DataFrame(processed_data)
            print(f"Dados carregados: {len(df)} registros de {len(df['device_id'].unique())} dispositivos")
            return df
            
        except Exception as e:
            print(f"Erro ao carregar dados do Firebase: {e}")
            return None
    
    def generate_synthetic_data(self, n_samples=1000):
        """
        Gera dados sintéticos para treinamento quando dados reais não estão disponíveis
        
        Args:
            n_samples (int): Número de amostras a gerar
            
        Returns:
            pd.DataFrame: DataFrame com dados sintéticos
        """
        np.random.seed(42)
        
        # Gerar dados sintéticos baseados em faixas realistas para água doce
        data = []
        
        for i in range(n_samples):
            # Definir classe de qualidade primeiro (0: Boa, 1: Regular, 2: Crítica)
            quality_class = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
            
            if quality_class == 0:  # Boa qualidade
                ph = np.random.normal(7.2, 0.3)
                turbidity = np.random.exponential(2.0)
                dissolved_oxygen = np.random.normal(8.5, 1.0)
                conductivity = np.random.normal(250, 50)
                temperature = np.random.normal(24, 3)
                
            elif quality_class == 1:  # Qualidade regular
                ph = np.random.choice([
                    np.random.normal(6.2, 0.2),  # Ligeiramente ácido
                    np.random.normal(8.8, 0.2)   # Ligeiramente básico
                ])
                turbidity = np.random.exponential(8.0)
                dissolved_oxygen = np.random.normal(6.0, 1.5)
                conductivity = np.random.normal(400, 100)
                temperature = np.random.normal(26, 4)
                
            else:  # Qualidade crítica
                ph = np.random.choice([
                    np.random.normal(5.5, 0.3),  # Muito ácido
                    np.random.normal(9.5, 0.3)   # Muito básico
                ])
                turbidity = np.random.exponential(20.0)
                dissolved_oxygen = np.random.normal(3.0, 1.0)
                conductivity = np.random.normal(800, 200)
                temperature = np.random.normal(30, 5)
            
            # Aplicar limites realistas
            ph = np.clip(ph, 4.0, 11.0)
            turbidity = np.clip(turbidity, 0.1, 100.0)
            dissolved_oxygen = np.clip(dissolved_oxygen, 0.5, 15.0)
            conductivity = np.clip(conductivity, 10, 2000)
            temperature = np.clip(temperature, 15, 40)
            
            data.append({
                'device_id': f'sensor_{i % 10 + 1:03d}',
                'timestamp': int((datetime.now() - timedelta(days=np.random.randint(0, 365))).timestamp() * 1000),
                'ph': round(ph, 2),
                'turbidity': round(turbidity, 2),
                'dissolved_oxygen': round(dissolved_oxygen, 2),
                'conductivity': round(conductivity, 1),
                'temperature': round(temperature, 1),
                'water_quality_class': quality_class
            })
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"Dados sintéticos gerados: {len(df)} registros")
        print(f"Distribuição de classes: {df['water_quality_class'].value_counts().to_dict()}")
        
        return df
    
    def classify_water_quality(self, df):
        """
        Classifica a qualidade da água baseada nos parâmetros medidos
        
        Args:
            df (pd.DataFrame): DataFrame com dados dos sensores
            
        Returns:
            pd.DataFrame: DataFrame com coluna de classificação adicionada
        """
        def classify_sample(row):
            score = 100
            
            # Avaliar pH (ideal: 6.5 - 8.5)
            if row['ph'] < 6.0 or row['ph'] > 9.0:
                score -= 30
            elif row['ph'] < 6.5 or row['ph'] > 8.5:
                score -= 15
            
            # Avaliar turbidez (ideal: < 5 NTU)
            if row['turbidity'] > 20:
                score -= 25
            elif row['turbidity'] > 5:
                score -= 10
            
            # Avaliar oxigênio dissolvido (ideal: > 5 mg/L)
            if row['dissolved_oxygen'] < 3:
                score -= 25
            elif row['dissolved_oxygen'] < 5:
                score -= 10
            
            # Avaliar condutividade (ideal: 50-500 µS/cm para água doce)
            if row['conductivity'] > 1000 or row['conductivity'] < 10:
                score -= 20
            elif row['conductivity'] > 500 or row['conductivity'] < 50:
                score -= 10
            
            # Classificar baseado no score
            if score >= 80:
                return 0  # Boa
            elif score >= 60:
                return 1  # Regular
            else:
                return 2  # Crítica
        
        if 'water_quality_class' not in df.columns:
            df['water_quality_class'] = df.apply(classify_sample, axis=1)
        
        return df
    
    def clean_data(self, df):
        """
        Limpa e valida os dados
        
        Args:
            df (pd.DataFrame): DataFrame com dados brutos
            
        Returns:
            pd.DataFrame: DataFrame limpo
        """
        # Remover linhas com valores ausentes nas colunas essenciais
        df_clean = df.dropna(subset=self.feature_columns)
        
        # Remover outliers extremos (usando IQR)
        for col in self.feature_columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        print(f"Dados após limpeza: {len(df_clean)} registros (removidos {len(df) - len(df_clean)} outliers)")
        
        return df_clean
    
    def create_features(self, df):
        """
        Cria features adicionais para o modelo
        
        Args:
            df (pd.DataFrame): DataFrame com dados básicos
            
        Returns:
            pd.DataFrame: DataFrame com features adicionais
        """
        df_features = df.copy()
        
        # Features derivadas
        df_features['ph_deviation'] = abs(df_features['ph'] - 7.0)  # Desvio do pH neutro
        df_features['oxygen_saturation_ratio'] = df_features['dissolved_oxygen'] / 10.0  # Normalizar OD
        df_features['conductivity_log'] = np.log1p(df_features['conductivity'])  # Log da condutividade
        
        # Features de interação
        df_features['ph_temp_interaction'] = df_features['ph'] * df_features['temperature']
        df_features['turbidity_conductivity_ratio'] = df_features['turbidity'] / (df_features['conductivity'] + 1)
        
        # Features temporais (se timestamp disponível)
        if 'datetime' in df_features.columns:
            df_features['hour'] = df_features['datetime'].dt.hour
            df_features['day_of_week'] = df_features['datetime'].dt.dayofweek
            df_features['month'] = df_features['datetime'].dt.month
        
        return df_features
    
    def prepare_for_training(self, df, test_size=0.2, validation_size=0.1):
        """
        Prepara os dados para treinamento
        
        Args:
            df (pd.DataFrame): DataFrame com dados processados
            test_size (float): Proporção dos dados para teste
            validation_size (float): Proporção dos dados para validação
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Selecionar features para o modelo
        feature_cols = self.feature_columns + [
            'ph_deviation', 'oxygen_saturation_ratio', 'conductivity_log',
            'ph_temp_interaction', 'turbidity_conductivity_ratio'
        ]
        
        # Adicionar features temporais se disponíveis
        if 'hour' in df.columns:
            feature_cols.extend(['hour', 'day_of_week', 'month'])
        
        X = df[feature_cols]
        y = df[self.target_column]
        
        # Dividir em treino e teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Dividir treino em treino e validação
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Dados preparados para treinamento:")
        print(f"  Treino: {X_train_scaled.shape[0]} amostras")
        print(f"  Validação: {X_val_scaled.shape[0]} amostras")
        print(f"  Teste: {X_test_scaled.shape[0]} amostras")
        print(f"  Features: {X_train_scaled.shape[1]}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def save_preprocessed_data(self, data_splits, output_dir):
        """
        Salva os dados pré-processados
        
        Args:
            data_splits (tuple): Dados divididos para treino/validação/teste
            output_dir (str): Diretório de saída
        """
        X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Salvar arrays numpy
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # Salvar scaler
        import joblib
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        
        print(f"Dados pré-processados salvos em: {output_dir}")
    
    def visualize_data(self, df, output_dir):
        """
        Cria visualizações dos dados
        
        Args:
            df (pd.DataFrame): DataFrame com dados processados
            output_dir (str): Diretório para salvar as visualizações
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        
        # 1. Distribuição das classes
        plt.figure(figsize=(8, 6))
        class_counts = df['water_quality_class'].value_counts()
        class_labels = ['Boa', 'Regular', 'Crítica']
        plt.pie(class_counts.values, labels=class_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Distribuição das Classes de Qualidade da Água')
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Matriz de correlação
        plt.figure(figsize=(10, 8))
        corr_matrix = df[self.feature_columns + ['water_quality_class']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação dos Parâmetros')
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Boxplots por classe
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(self.feature_columns):
            sns.boxplot(data=df, x='water_quality_class', y=col, ax=axes[i])
            axes[i].set_title(f'Distribuição de {col} por Classe')
            axes[i].set_xticklabels(['Boa', 'Regular', 'Crítica'])
        
        # Remover subplot extra
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizações salvas em: {output_dir}")

def main():
    """
    Função principal para executar o pré-processamento
    """
    preprocessor = WaterQualityPreprocessor()
    
    # Tentar carregar dados reais do Firebase (se disponível)
    firebase_data_path = '../data/firebase_export.json'
    
    if os.path.exists(firebase_data_path):
        print("Carregando dados do Firebase...")
        df = preprocessor.load_data_from_firebase_export(firebase_data_path)
    else:
        print("Dados do Firebase não encontrados. Gerando dados sintéticos...")
        df = preprocessor.generate_synthetic_data(n_samples=2000)
    
    if df is None or len(df) == 0:
        print("Erro: Nenhum dado disponível para processamento.")
        return
    
    # Classificar qualidade da água (se não estiver classificado)
    df = preprocessor.classify_water_quality(df)
    
    # Limpar dados
    df_clean = preprocessor.clean_data(df)
    
    # Criar features adicionais
    df_features = preprocessor.create_features(df_clean)
    
    # Preparar dados para treinamento
    data_splits = preprocessor.prepare_for_training(df_features)
    
    # Salvar dados pré-processados
    output_dir = '../data/processed'
    preprocessor.save_preprocessed_data(data_splits, output_dir)
    
    # Criar visualizações
    viz_dir = '../data/visualizations'
    preprocessor.visualize_data(df_features, viz_dir)
    
    print("Pré-processamento concluído com sucesso!")

if __name__ == "__main__":
    main()

