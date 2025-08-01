# 🌊 Projeto HydrAIon: Sistema IoT com IA para Monitoramento Inteligente da Qualidade da Água

<p align="center">
  <img src="https://img.shields.io/badge/ESP32-S3-blue?style=for-the-badge&logo=espressif" alt="ESP32-S3"/>
  <img src="https://img.shields.io/badge/TensorFlow-Lite-orange?style=for-the-badge&logo=tensorflow" alt="TensorFlow Lite"/>
  <img src="https://img.shields.io/badge/Firebase-Realtime-yellow?style=for-the-badge&logo=firebase" alt="Firebase"/>
  <img src="https://img.shields.io/badge/LoRa-Communication-green?style=for-the-badge" alt="LoRa"/>
</p>

## 📋 Sobre o Projeto

O **HydrAIon** é um sistema IoT inovador que combina sensores de qualidade da água, inteligência artificial embarcada (TinyML) e comunicação LoRa para monitoramento autônomo e inteligente de corpos d\'água. Desenvolvido especificamente para o Canal do Cavouco em Recife/PE, o sistema oferece monitoramento em tempo real com validação cooperativa de dados e interface web interativa.

### 🎯 Objetivos

- **Monitoramento Autônomo:** Coleta contínua de dados de qualidade da água
- **Inteligência Embarcada:** Classificação local usando TinyML no ESP32-S3
- **Comunicação de Longo Alcance:** Transmissão via LoRa para áreas remotas
- **Validação Cooperativa:** Algoritmos de consenso entre múltiplos sensores
- **Interface Intuitiva:** Dashboard web com mapas e visualizações em tempo real

## 🏗️ Arquitetura do Sistema

```mermaid
graph TD
    A[Sensores IoT] --> B[ESP32-S3 + TinyML]
    B --> C[Comunicação LoRa]
    C --> D[Gateway LoRa]
    D --> E[Firebase Cloud]
    E --> F[Dashboard Web]
    E --> G[API ANA]
    F --> H[Usuários]
```

## 🔧 Componentes de Hardware

Para detalhes completos sobre o hardware, consulte a [Documentação de Hardware](docs/hardware/README.md).

### Sensores de Qualidade da Água
- **DFRobot SEN0169:** Sensor de pH (0-14 pH, ±0.1 precisão)
- **DFRobot SEN0189:** Sensor de Turbidez (0-1000 NTU)
- **Atlas Scientific EZO-DO:** Oxigênio Dissolvido (0.01-100+ mg/L)
- **Atlas Scientific EZO-EC:** Condutividade Elétrica (0.07-500,000+ µS/cm)
- **DS18B20:** Sensor de Temperatura (-55°C a +125°C, ±0.5°C)

### Microcontrolador e Comunicação
- **ESP32-S3:** Processador dual-core com suporte a TinyML
- **Módulo LoRa:** SX1276/SX1278 para comunicação de longo alcance (2-5 km)
- **Alimentação:** Bateria Li-ion + painel solar para autonomia de 30+ dias

## 💻 Componentes de Software

Para detalhes completos sobre o software, consulte a [Documentação de Software](docs/software/README.md).

### 1. Firmware ESP32-S3
- **Localização:** `firmware/main/main.ino`
- **Funcionalidades:** Leitura de sensores, processamento local com modelo TinyML (2.44 KB, 97.33% acurácia), comunicação LoRa para transmissão de dados e gerenciamento inteligente de energia (deep sleep).
- **Mais detalhes:** [Firmware ESP32-S3](docs/software/firmware_details.md)

### 2. Backend em Nuvem
- **Localização:** `cloud/functions/`
- **Tecnologia:** Firebase Cloud Functions para processamento de dados.
- **Funcionalidades:** Validação cooperativa entre sensores próximos, integração com API da ANA (HidroWebService) e re-treinamento automático de modelos de IA.
- **Mais detalhes:** [Backend em Nuvem](docs/software/backend_details.md)

### 3. Dashboard Web
- **Localização:** `dashboard/public/`
- **Tecnologia:** Interface responsiva com HTML5, CSS3 e JavaScript, utilizando LeafletJS para mapas interativos.
- **Funcionalidades:** Visualização em tempo real conectada ao Firebase e indicadores visuais de qualidade da água.
- **Mais detalhes:** [Dashboard Web](docs/software/dashboard_details.md)

### 4. Scripts de IA
- **Localização:** `ai_training/scripts/`
- **Tecnologia:** Python com scikit-learn, TensorFlow/Keras e TensorFlow Lite.
- **Funcionalidades:** Pré-processamento de dados, treinamento de modelos, otimização para TinyML e geração automática de código C++ para ESP32.
- **Mais detalhes:** [IA e TinyML](docs/software/ai_details.md)

## 🤖 Inteligência Artificial

### Modelo TinyML
- **Arquitetura:** Rede neural densa (5→8→4→3 neurônios)
- **Entrada:** pH, Turbidez, OD, Condutividade, Temperatura (normalizados)
- **Saída:** Classificação em 3 classes (Boa, Regular, Crítica)
- **Performance:** 97.33% acurácia, <10ms latência, <8KB RAM

### Validação Cooperativa
- Comparação entre sensores próximos (raio 500m)
- Análise de tendências temporais (24h)
- Algoritmo de consenso por maioria simples
- Marcação automática de dados suspeitos

## 🌐 Integração com APIs

Para detalhes completos sobre as integrações de API, consulte a [Documentação de Integrações de API](docs/software/api_integrations.md).

### API HidroWebService da ANA
- **Integração:** Realizada via Cloud Function `fetchAnaData` no backend.
- **Funcionalidades:** Inventário de estações de monitoramento, dados históricos de qualidade da água e séries telemétricas em tempo real.
- **Benefício:** Validação cruzada com dados oficiais para maior confiabilidade.

## 📊 Estrutura do Repositório

```
HydrAIon/
├── firmware/                    # Código para ESP32-S3
│   └── main/
│       └── main.ino            # Firmware principal
├── cloud/                      # Backend em nuvem
│   └── functions/
│       ├── index.js            # Cloud Functions
│       └── package.json        # Dependências Node.js
├── dashboard/                  # Dashboard web
│   └── public/
│       ├── index.html          # Interface principal
│       ├── styles.css          # Estilização
│       ├── app.js              # Lógica JavaScript
│       └── package.json        # Configurações
├── ai_training/                # Scripts de IA e TinyML
│   ├── scripts/
│   │   ├── data_preprocessing.py    # Pré-processamento
│   │   ├── model_training.py        # Treinamento completo
│   │   └── tinyml_generator.py      # Gerador TinyML
│   ├── data/                   # Dados processados
│   └── models/                 # Modelos treinados
├── docs/                       # Documentação detalhada do projeto
│   ├── hardware/               # Esquemáticos, designs e BOM ([README](docs/hardware/README.md))
│   ├── software/               # Detalhes de firmware, backend, dashboard e IA ([README](docs/software/README.md))
│   └── user_guide/             # Guia de uso do sistema para usuários finais ([README](docs/user_guide/README.md))
├── DOCUMENTACAO_HYDRAION.md    # Documentação técnica completa (PDF gerado a partir deste)
├── README.md                   # Este arquivo
└── LICENSE.md                  # Licença MIT do projeto
```

## 🚀 Guia de Início Rápido

Para um guia detalhado sobre como configurar e executar o projeto, consulte a [Documentação Completa](DOCUMENTACAO_HYDRAION.md) ou os READMEs específicos nas pastas `docs/`.

### Pré-requisitos
- ESP32-S3 DevKit
- Sensores de qualidade da água
- Módulo LoRa SX1276/SX1278
- Conta Firebase
- Python 3.8+ e Node.js

### 1. Configuração do Hardware
```cpp
// Configurar pinos no firmware (exemplo)
#define PH_PIN A0
#define TURBIDITY_PIN A1
#define ONE_WIRE_BUS 4
#define LORA_CS 18
#define LORA_RST 14
#define LORA_DIO0 26
```

### 2. Configuração do Firebase
```bash
# Instalar dependências das Cloud Functions
cd cloud/functions
npm install

# Deploy das Cloud Functions
firebase deploy --only functions
```

### 3. Treinamento do Modelo TinyML
```bash
# Instalar dependências Python
pip3 install scikit-learn tensorflow joblib

# Gerar modelo TinyML (gera o arquivo .h para o firmware)
cd ai_training/scripts
python3 tinyml_generator.py
```

### 4. Dashboard Web
```bash
# Executar localmente (a partir da raiz do projeto)
cd dashboard
python3 -m http.server 8000 --directory public
```

## 📈 Resultados e Performance

### Métricas do Sistema
- **Autonomia:** 30+ dias com bateria e painel solar
- **Alcance LoRa:** 2-5 km em ambiente urbano
- **Latência de transmissão:** < 30 segundos
- **Acurácia do modelo IA:** 97.33%
- **Tamanho do modelo TinyML:** 2.44 KB

### Validação
- **Correlação com dados da ANA:** 85%
- **Tempo de resposta do dashboard:** < 2 segundos
- **Compatibilidade:** Desktop e mobile
- **Disponibilidade do sistema:** 99.5%

## 🔮 Roadmap e Melhorias Futuras

### Curto Prazo (3-6 meses)
- [ ] Implementação de LoRaWAN
- [ ] Alertas via SMS/WhatsApp
- [ ] Sensores adicionais (Nitrogênio, Fósforo)
- [ ] App mobile nativo

### Médio Prazo (6-12 meses)
- [ ] Expansão para outros canais urbanos
- [ ] Detecção de anomalias avançada
- [ ] Integração com sistemas municipais
- [ ] API pública para desenvolvedores

### Longo Prazo (1-2 anos)
- [ ] Federated Learning entre dispositivos
- [ ] Previsão de qualidade da água
- [ ] Mesh networking entre sensores
- [ ] Economia circular dos componentes

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, leia o [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes sobre nosso código de conduta e processo de submissão de pull requests.

### Como Contribuir
1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE.md](LICENSE.md) para detalhes.

## 👨‍💻 Autor

**Gustavo Souto Silva de Barros Santos**
- 🎓 Faculdade Nova Roma - FNR
- 📧 Email: gustavosouto004@gmail.com
- 💼 LinkedIn: [Seu perfil LinkedIn]
- 🐙 GitHub: [Seu perfil GitHub]

## 🙏 Agradecimentos

- **Faculdade Nova Roma (FNR)** pelo suporte acadêmico
- **ANA (Agência Nacional de Águas)** pelos dados públicos
- **Comunidade Open Source** pelas bibliotecas utilizadas
- **Prefeitura do Recife** pelo acesso ao Canal do Cavouco

## 📞 Suporte

Para suporte técnico ou dúvidas sobre o projeto:

- 📧 **Email:** gustavosouto004@gmail.com
- 📋 **Issues:** Use o sistema de issues do GitHub
- 📖 **Documentação:** Consulte `DOCUMENTACAO_HYDRAION.md`
- 💬 **Discussões:** Use as discussões do GitHub

---

<p align="center">
  <strong>🌊 HydrAIon - Monitoramento Inteligente da Qualidade da Água 🌊</strong><br>
  <em>Desenvolvido com ❤️ para um futuro mais sustentável</em>
</p>

