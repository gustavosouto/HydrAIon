# 🌊 Projeto HydrAIon: Sistema IoT com IA para Monitoramento Inteligente da Qualidade da Água

<p align="center">
  <img src="https://placehold.co/800x400/007BFF/FFFFFF?text=Imagem+do+Sensor+HydrAIon+em+Operação" alt="Imagem conceitual do protótipo HydrAIon em ambiente aquático urbano">
  <br>
  <em>**IMPORTANTE: Substitua esta imagem pela foto real do seu protótipo HydrAIon em ambiente aquático urbano para uma apresentação mais impactante.**</em>
</p>

## 🔬 Visão Geral do Projeto

O **Projeto HydrAIon** surge como uma resposta inovadora e tecnológica à crescente necessidade de monitoramento ambiental em ecossistemas hídricos urbanos. Trata-se do desenvolvimento de um **sistema inteligente, cooperativo e autônomo** de Internet das Coisas (IoT ) integrado com Inteligência Artificial (IA) para o **monitoramento contínuo da qualidade da água** em rios e canais.

Focado na Região Metropolitana do Recife (RMR), onde a poluição hídrica é uma problemática latente e de grande impacto socioambiental, o HydrAIon visa superar as limitações dos métodos de monitoramento convencionais e das soluções existentes, tais como conectividade intermitente, elevado consumo energético e dificuldades operacionais de manutenção em ambientes fluviais.

Através de uma arquitetura robusta baseada em **redes mesh LoRa** e **inferência local via TinyML**, o sistema proposto busca fornecer dados ambientais em tempo real com **alta confiabilidade, acurácia e eficiência energética**, garantindo uma autonomia operacional contínua estimada em até 12 meses.

### Objetivos Educacionais (Metodologia PBL - Fase I)

Este projeto, concebido sob a metodologia de Aprendizagem Baseada em Projetos (PBL), possui objetivos educacionais específicos para a Fase I, visando o desenvolvimento prático e teórico da equipe:

*   **Compreensão Aprofundada:** Entender o funcionamento de sensores ambientais e microcontroladores em aplicações de campo.
*   **Medição Precisa:** Habilitar a medição acurada de parâmetros críticos da qualidade da água (pH, turbidez, temperatura, oxigênio dissolvido e condutividade).
*   **Aplicação IoT:** Dominar conceitos de IoT, desde a coleta até o envio seguro de dados para plataformas em nuvem.
*   **Desenvolvimento de IA:** Capacitar no desenvolvimento e implementação de modelos de IA para análise preditiva, detecção de anomalias e classificação da qualidade da água.
*   **Trabalho Colaborativo:** Fomentar o trabalho em equipe, seguindo os princípios da metodologia PBL.

## ✨ Recursos e Funcionalidades

O sistema HydrAIon é dotado de um conjunto de funcionalidades estratégicas para um monitoramento eficaz e proativo:

*   **Coleta de Dados Contínua:** Módulos sensores (flutuantes e fixos) equipados para aquisição ininterrupta de dados físico-químicos da água.
*   **Inteligência Artificial Embarcada (TinyML):** Modelos leves de machine learning executados diretamente nos microcontroladores (ESP32-S3), permitindo a classificação preliminar da qualidade da água ("bom", "regular", "crítico") e detecção de anomalias localmente, reduzindo a transmissão de dados brutos e economizando energia.
*   **Rede Mesh LoRa Resiliente:** Topologia de rede em malha que assegura comunicação de longo alcance e baixo consumo entre os nós sensores e os gateways, provendo redundância e reconfiguração dinâmica de rotas em ambientes com obstáculos.
*   **Validação Cooperativa de Dados:** Algoritmos de consenso distribuído verificam a consistência das medições entre múltiplos sensores próximos, elevando a confiabilidade dos dados reportados e mitigando falsos positivos.
*   **Armazenamento e Processamento em Nuvem:** Utilização do Firebase Realtime Database para centralização, organização e versionamento dos dados coletados, com alta disponibilidade e segurança. A nuvem suporta o re-treinamento incremental dos modelos de IA e a distribuição de atualizações de firmware (OTA).
*   **Dashboard Interativo Georreferenciado:** Plataforma de visualização desenvolvida com LeafletJS, apresentando os pontos de monitoramento sobre mapas interativos com indicadores visuais de qualidade da água (escala cromática intuitiva).
*   **Sistema de Alertas Automatizados:** Integração com a API do Telegram para envio instantâneo de notificações a autoridades ambientais e comunidades impactadas em caso de detecção de anomalias ou condições críticas da água.
*   **Exportação de Dados:** Funcionalidade para extração de dados históricos em formatos padronizados (CSV, JSON), facilitando análises externas e o acervo institucional.
*   **Autonomia Energética Avançada:** Sistema de alimentação híbrido com células fotovoltaicas de alta eficiência e baterias de lítio-ferro-fosfato (LiFePO4) com sistema de gerenciamento (BMS), garantindo operação contínua por um período prolongado.

## 📊 Parâmetros Monitorados

O HydrAIon monitora um conjunto essencial de parâmetros da qualidade da água, cada um fornecendo informações críticas para a avaliação ambiental:

| Parâmetro                 | Descrição Detalhada                                        | Importância para a Qualidade da Água                                       |
| :------------------------ | :--------------------------------------------------------- | :------------------------------------------------------------------------- |
| **pH** | Mede o potencial hidrogeniônico, indicando a acidez ou basicidade da água. | Influencia a solubilidade de nutrientes e poluentes, e a sobrevivência de organismos aquáticos. |
| **Turbidez** | Avalia a presença de partículas suspensas (argila, silte, matéria orgânica). | Indica a claridade da água e pode sinalizar poluição por esgoto ou erosão, afetando a penetração da luz. |
| **Oxigênio Dissolvido (OD)** | Quantidade de oxigênio gasoso dissolvido na água.          | Essencial para a respiração de peixes e outros organismos aquáticos; baixos níveis indicam poluição orgânica. |
| **Temperatura** | Medida da energia térmica da água.                         | Afeta a solubilidade de gases (como OD), a taxa metabólica de organismos e a toxicidade de poluentes. |
| **Condutividade Elétrica** | Capacidade da água de conduzir corrente elétrica.          | Indicador da concentração total de íons dissolvidos; altos valores podem sugerir contaminação. |

## 📐 Arquitetura do Sistema

A arquitetura do HydrAIon é modular e distribuída, projetada para robustez e eficiência. Ela integra hardware e software em um ecossistema coeso:

*   **Módulos Sensores Inteligentes:** Unidades autônomas (flutuantes ou fixas) com sensores, microcontrolador ESP32-S3 e IA embarcada para coleta e pré-análise de dados.
*   **Rede de Comunicação Mesh LoRa:** Conecta os módulos sensores, permitindo a retransmissão de dados entre eles para ampliar o alcance e a resiliência da rede.
*   **Gateways de Agregação:** Dispositivos que coletam dados da rede LoRa e os enviam para a nuvem via conectividade 4G/5G ou Wi-Fi.
*   **Serviço de Validação e Consenso:** Lógica de software que cruza dados de múltiplos sensores para garantir a integridade e veracidade das informações antes do armazenamento central.
*   **Camada de Armazenamento e Processamento em Nuvem:** Base de dados (Firebase) para persistência e gestão dos dados, com capacidade de processamento para re-treinamento de modelos de IA e atualizações OTA.
*   **Dashboard e Alertas:** Interface de usuário para visualização em tempo real e sistema de notificação automática.

**(Para o diagrama visual da arquitetura, consulte a página 4 do documento "Projeto Fase I - HydrAIon Sistema IoT com IA para Monitoramento da Qualidade da Água.pdf" no diretório `docs/` deste repositório.)**

## ⚙️ Tecnologias-Chave Utilizadas

Esta seção lista as **tecnologias, frameworks e plataformas de software e hardware** que formam a espinha dorsal do projeto HydrAIon, com justificativas para suas escolhas.

| Categoria                    | Tecnologia/Plataforma                  | Detalhes e Justificativa da Escolha                                                                  |
| :--------------------------- | :--------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| **Microcontrolador** | ESP32-S3                                 | Selecionado por seu baixo consumo, capacidade de processamento para TinyML e conectividade Wi-Fi/Bluetooth, ideal para prototipagem e ambientes de baixa energia. |
| **Comunicação IoT** | LoRa, Rede Mesh, Protocolo MQTT/MQTT-SN  | Garante conectividade robusta, de longo alcance e baixo consumo em ambientes urbanos com obstáculos, otimizando o transporte de dados. |
| **Inteligência Artificial** | TinyML                                   | Permite a execução de modelos de IA leves diretamente no dispositivo (inferência local), reduzindo a latência e o volume de dados transmitidos. |
| **Backend & Nuvem** | Firebase Realtime Database, Google Cloud Functions | Plataforma NoSQL em tempo real que facilita o armazenamento, sincronização e gestão de dados, além de suportar a orquestração de IA. |
| **Visualização de Dados** | LeafletJS                                | Biblioteca JavaScript para mapas interativos e responsivos, essencial para a visualização georreferenciada dos pontos de monitoramento. |
| **Sistema de Alertas** | Telegram API                             | Escolhida pela facilidade de integração e ampla adoção, permitindo o envio rápido e automatizado de notificações críticas. |
| **Gerenciamento de Energia** | Controlador MPPT, BMS (Battery Management System) | Maximizam a eficiência da captação solar e garantem a segurança e longevidade das baterias, cruciais para a autonomia do sistema. |
| **Proteção Física** | Grau de Proteção IP67                    | Padrão internacional que assegura resistência à imersão temporária em água e proteção total contra poeira, fundamental para ambientes aquáticos. |
| **Metodologia de Desenvolvimento** | PBL (Problem-Based Learning)             | Abordagem que estimula a aprendizagem ativa, o trabalho em equipe e a resolução de problemas reais, alinhando os objetivos educacionais aos técnicos do projeto. |

## ⚡ Componentes Eletrônicos Detalhados

Esta seção descreve os **componentes físicos (hardware) específicos** que serão integrados no protótipo HydrAIon, com suas respectivas funções e justificativas.

| Componente                | Modelo/Tipo Específico                       | Função Principal                                          | Justificativa da Escolha para a Fase I                                     |
| :------------------------ | :------------------------------------------- | :-------------------------------------------------------- | :------------------------------------------------------------------------- |
| **Microcontrolador** | ESP32-S3                                     | Cérebro central dos módulos sensores, responsável pelo processamento e controle. | Equilíbrio entre custo, poder de processamento para TinyML e facilidade de integração em protótipos. |
| **Sensores de Qualidade da Água** |                                          |                                                           |                                                                            |
| &nbsp;&nbsp; pH          | DFRobot SEN0169 (industrial, selado com ATC) | Sonda robusta e precisa para medição do potencial hidrogeniônico da água. | Robusto, selado e com compensação de temperatura, ideal para durabilidade e precisão em ambiente aquático. |
| &nbsp;&nbsp; Turbidez     | DFRobot SEN0189                              | Detecção de sólidos em suspensão na água.                 | Solução acessível para prototipagem; demanda atenção à bioincrustação com planos de limpeza. |
| &nbsp;&nbsp; OD           | Atlas Scientific EZO-DO™ (ótico)             | Avaliação da quantidade de oxigênio dissolvido.           | Oferece boa robustez e precisão para protótipos, com menor manutenção que sensores eletroquímicos. |
| &nbsp;&nbsp; Condutividade | Atlas Scientific EZO-EC™                     | Indicação da presença de íons e potenciais agentes contaminantes. | Proporciona leituras confiáveis e sua interface simplifica a integração com o microcontrolador. |
| &nbsp;&nbsp; Temperatura  | DS18B20 (em cápsula de aço inoxidável)       | Medição da temperatura da água.                           | Alta resistência à água, precisão adequada e facilidade de implementação para contextualização dos dados. |
| **Módulos de Comunicação** |                                          |                                                           |                                                                            |
| &nbsp;&nbsp; LoRa         | ESP32 com LoRa integrado (Heltec/LilyGo)     | Permitem a comunicação de longo alcance e baixo consumo entre os nós da rede e o gateway. | Simplificam a montagem e o desenvolvimento ao integrar microcontrolador e módulo LoRa em uma única placa. |
| **Sistema de Alimentação** |                                          |                                                           |                                                                            |
| &nbsp;&nbsp; Célula Fotovoltaica | Monocristalina de alta eficiência            | Componente para captação de energia solar e recarga das baterias. | Alta taxa de conversão de luz solar em eletricidade em um formato compacto. |
| &nbsp;&nbsp; Bateria      | LiFePO4 (com BMS)                            | Armazenamento de energia para garantir a autonomia contínua do sistema. | Oferecem maior ciclo de vida, segurança e estabilidade térmica em comparação com Li-Ion, ideal para uso externo. |

## 🗓️ Cronograma de Desenvolvimento (Fase I - 5 meses)

A Fase I do projeto HydrAIon segue um cronograma estruturado, garantindo uma progressão lógica e a entrega de marcos importantes:

1.  **Mês 1: Planejamento e Arquitetura do Sistema**
    *   Levantamento detalhado de requisitos técnicos.
    *   Definição da arquitetura híbrida (sensores flutuantes e fixos).
    *   Seleção criteriosa dos sensores ambientais e do microcontrolador.
    *   Definição dos parâmetros de coleta e critérios de desempenho do sistema.
2.  **Mês 2: Prototipagem e Testes Iniciais**
    *   Montagem e prototipagem física dos módulos sensoriais (invólucro, fixação, eletrônica).
    *   Implementação do firmware inicial e dos modelos de TinyML para inferência local.
    *   Realização de testes laboratoriais rigorosos dos sensores para aferição de precisão e estabilidade.
3.  **Mês 3: Comunicação Mesh e Validação Cooperativa**
    *   Desenvolvimento e integração dos módulos à rede mesh LoRa.
    *   Implementação do protocolo MQTT otimizado para baixo consumo energético.
    *   Condução de simulações de falhas e testes de validação cruzada entre nós sensoriais com algoritmos de consenso.
4.  **Mês 4: Integração com Nuvem e Interface de Visualização**
    *   Desenvolvimento da comunicação segura entre os gateways e o ambiente em nuvem (Firebase).
    *   Criação do painel interativo georreferenciado utilizando LeafletJS.
    *   Implementação dos mecanismos de alertas automatizados via Telegram API.
    *   Configuração inicial para re-treinamento de modelos de IA e atualizações OTA.
5.  **Mês 5: Validação em Campo e Consolidação dos Resultados**
    *   Instalação e implantação dos protótipos no Canal do Cavouco, Recife/PE.
    *   Realização de testes operacionais em condições reais, incluindo simulações de cenários de anomalia.
    *   Aferição da resposta do sistema e coleta de dados de validação.
    *   Elaboração do relatório técnico final, consolidação de todos os dados e produção do vídeo demonstrativo do projeto.

## 📍 Local de Instalação do Protótipo

O protótipo do HydrAIon será instalado no **Canal do Cavouco, localizado em Recife/PE**. A escolha estratégica deste corpo hídrico se justifica por múltiplos fatores:

*   **Representatividade Ambiental:** O Canal do Cavouco apresenta características de um ambiente aquático urbano típico na RMR, com condições que permitem a avaliação do sistema em cenários reais de fluxo hídrico contínuo e presença de resíduos flutuantes.
*   **Localização Estratégica:** Sua posição em área urbana facilita o acesso para monitoramento, manutenção e testes operacionais.
*   **Interação com a População e Órgãos Públicos:** O local propicia a interação com a comunidade do entorno e as autoridades ambientais, fomentando a adesão ao monitoramento participativo e a coleta de feedback para futuras otimizações do sistema.
*   **Avaliação de Resiliência:** Permite testar a resistência estrutural dos módulos frente às intempéries e à presença de detritos, validando o design físico.

## 📂 Estrutura do Repositório

A estrutura do repositório GitHub para o HydrAIon seguirá uma organização lógica para facilitar o desenvolvimento, a colaboração e a documentação:

*   `firmware/`
    *   `src/` - Código-fonte embarcado para ESP32 (sensores e gateways)
    *   `lib/` - Bibliotecas
    *   `platformio.ini` - Configurações do PlatformIO
*   `cloud/`
    *   `functions/` - Funções para a plataforma Firebase (e Google Cloud Functions)
    *   `database-rules/` - Regras do Firebase Realtime Database
*   `dashboard/`
    *   `public/` - Arquivos públicos do painel de visualização (HTML, CSS, JS)
    *   `src/` - Código-fonte do painel de visualização (LeafletJS)
    *   `package.json` - Dependências do projeto
*   `docs/` - Documentação detalhada, relatórios técnicos e especificações
    *   `relatorio_fase_1.pdf` - Relatório da Fase 1
    *   `especificacoes_hardware.md` - Especificações de hardware
*   `hardware/` - Esquemáticos, designs de PCB e modelos 3D dos invólucros
    *   `esquematicos/` - Esquemáticos eletrônicos
    *   `invóluculos/` - Modelos 3D dos invólucros
*   `models/` - Modelos de IA (TinyML) para treinamento e inferência
    *   `tinyML_models/` - Modelos TinyML
    *   `training_scripts/` - Scripts de treinamento
*   `assets/` - Imagens, vídeos e outros recursos visuais do projeto
*   `.github/` - Configurações do GitHub (workflows, templates)
*   `.gitignore` - Arquivos e pastas a serem ignoradas pelo Git
*   `README.md` - Este arquivo
*   `LICENSE.md` - Informações sobre a licença do projeto

## 🚀 Guia de Início Rápido e Contribuição

Detalhes sobre como configurar o ambiente de desenvolvimento, compilar o firmware, interagir com a nuvem e contribuir para o projeto serão adicionados em futuras versões deste README, à medida que o desenvolvimento avança.

Para começar, você pode clonar o repositório:

```bash
git clone https://github.com/seu-usuario/hydraion.git # Substitua 'seu-usuario' pelo nome de usuário/organização real
cd hydraion
